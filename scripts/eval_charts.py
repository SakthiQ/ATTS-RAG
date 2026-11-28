# -*- coding: utf-8 -*-
"""Measures Layer 3's chart check (verify_chart) and chart trigger (wants_chart) on tests/data/chart_cases.json.

    python scripts/eval_charts.py --out eval/results/charts_v2/results.json             # current check
    python scripts/eval_charts.py --out eval/results/charts_v1/results.json --chart-v1  # the check before v2
    python scripts/eval_charts.py ... --models llama3 gemma4:e4b                        # + the model proposes charts
    python scripts/eval_charts.py ... --skip-e2e                                        # no model calls

Three parts:
  * cases: hand-made chart proposals (correct ones and wrong ones of each kind) are checked;
  * wants_chart: does the trigger fire when a chart can help, and stay quiet when it cannot;
  * e2e: a local model proposes a chart for each question, as in the app (same prompt, JSON mode, 4,096-token context,
    at most 300 tokens), then the check decides.

Ground truth comes from an oracle in this file, independent of the code under test, and it judges what is RELEASED
(after a pie or line chart is shown as a bar chart, or a line is put in order). A released chart is correct only if all
its points are one series of the evidence: one table column (rows -> values) or one table row across same-unit columns
(column names -> values) or the figures of one sentence; a pie only for parts of one whole (shares adding up to 100, or
all the parts of a Total row); a line only over time-like labels in the document's order; never a Total next to its
parts; list numbering is never a value; at least 2 points, no repeated label, every value exactly as written.
  precision = released charts that are correct / released charts
  recall    = correct proposals whose released chart is correct / correct proposals
"""
import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from loguru import logger  # noqa: E402

from app.rag.layer3 import generation  # noqa: E402
from app.rag.layer3.contracts import ChartPoint, ChartSpec  # noqa: E402

DATA_PATH = os.path.join("tests", "data", "chart_cases.json")
_NUM = re.compile(r"^\s*([0-9][0-9,]*\.?[0-9]*)\s*(M|%|USD)?\s*$", re.I)
_TIME = re.compile(r"^(q[1-4]( (19|20)\d\d)?|(19|20)\d\d|h[12]|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|week \d+)", re.I)
_NUMBERING = re.compile(r"^(phase|step|pitfall|wave|no\.?|#|rank)$", re.I)


def _num(cell: str) -> Optional[Tuple[float, str]]:
    m = _NUM.match(cell)
    return (float(m.group(1).replace(",", "")), (m.group(2) or "").upper()) if m else None


def _norm(label: str) -> str:
    """Lower case, '-' as a space, and no '#' ("Pitfall #1" names the same row as "Pitfall 1")."""
    return " ".join(label.lower().replace("-", " ").replace("#", " ").split())


def table_series(pid: str, text: str) -> List[dict]:
    """Every one-column and one-row series of the Markdown table in `text`."""
    lines = [l.strip() for l in text.splitlines() if l.strip().startswith("|")]
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = [[c.strip() for c in l.strip("|").split("|")] for l in lines[2:]]
    ncol = len(header)
    numeric = {j for j in range(ncol) if all(_num(r[j]) for r in rows)}
    numbering = {j for j in numeric if [_num(r[j])[0] for r in rows] == list(range(1, len(rows) + 1))
                 and (j == 0 or _NUMBERING.match(header[j]))}
    value_cols = sorted(numeric - numbering)
    label_cols = [j for j in range(ncol) if j not in numeric and len({r[j] for r in rows}) == len(rows)]
    if not label_cols:
        return []
    lab = label_cols[0]
    series = []
    total_rows = [r for r in rows if r[lab].lower() == "total"]
    parts = [r for r in rows if r[lab].lower() != "total"]
    for j in value_cols:
        aliases = []
        for r in parts:
            a = {_norm(r[lab])}
            for nj in numbering:   # "Phase 1" names the same row as its description
                a.add(_norm(f"{header[nj]} {r[nj]}"))
            aliases.append(a)
        values = [_num(r[j])[0] for r in parts]
        unit = _num(parts[0][j])[1]
        whole = bool(total_rows and abs(sum(values) - _num(total_rows[0][j])[0]) < 1e-6) or \
            (("%" in header[j] or unit == "%") and abs(sum(values) - 100) <= 1)
        series.append({"passage": pid, "kind": f"column '{header[j]}'", "aliases": aliases, "values": values,
                       "whole": whole, "ordered": all(_TIME.match(_norm(r[lab])) for r in parts)})
    for r in rows:   # one row across same-unit columns, ordered when the column names are quarters, years ...
        by_unit: Dict[str, List[int]] = {}
        for j in value_cols:
            by_unit.setdefault(_num(r[j])[1], []).append(j)
        for unit, cols in by_unit.items():
            if len(cols) >= 2:
                series.append({"passage": pid, "kind": f"row '{r[lab]}'", "aliases": [{_norm(header[j])} for j in cols],
                               "values": [_num(r[j])[0] for j in cols], "whole": False,
                               "ordered": all(_TIME.match(header[j]) for j in cols)})
    return series


def all_series(data: dict) -> List[dict]:
    out = []
    for pid, text in data["passages"].items():
        if "|" in text:
            out.extend(table_series(pid, text))
    for s in data["prose_series"]:
        out.append({"passage": s["passage"], "kind": "sentence", "aliases": [{_norm(l)} for l in s["labels"]],
                    "values": s["values"], "whole": s["whole"], "ordered": s["ordered"]})
    return out


def oracle(spec: ChartSpec, allowed: List[str], series: List[dict]) -> Tuple[bool, str]:
    pts = spec.data
    if len(pts) < 2:
        return False, "fewer than 2 points"
    labels = [_norm(p.label) for p in pts]
    if len(set(labels)) != len(labels):
        return False, "repeated label"
    if any(l == "total" for l in labels):
        return False, "Total shown next to its parts"
    reason = "points are not one series of the evidence (mixed columns, rows, tables or wrong values)"
    for s in series:
        if s["passage"] not in allowed:
            continue
        positions = []
        for p, l in zip(pts, labels):
            hit = [i for i, a in enumerate(s["aliases"]) if l in a and abs(s["values"][i] - p.value) < 1e-6]
            if not hit:
                break
            positions.append(hit[0])
        else:
            if spec.type == "pie" and not s["whole"]:
                reason = "pie of values that are not parts of one whole"
                continue
            if spec.type == "pie" and len(set(positions)) != len(s["values"]):
                reason = "pie that leaves out some of the parts"
                continue
            if spec.type == "line" and (not s["ordered"] or positions != sorted(positions)):
                reason = "line chart not in the document's order" if s["ordered"] else "line chart over unordered categories"
                continue
            return True, f"one series: {s['passage']} {s['kind']}"
    return False, reason


def make_spec(type_: str, pts, title: str = "Chart") -> ChartSpec:
    return ChartSpec(type=type_, title=title, data=[ChartPoint(label=l, value=v) for l, v in pts])


def lookup_for(data: dict, pids: List[str]) -> Dict[str, str]:
    return {f"E{i + 1}": data["passages"][p] for i, p in enumerate(pids)}


def scores(rows: List[dict]) -> dict:
    """rows: {'proposal_correct', 'released', 'released_correct'}."""
    released = [r for r in rows if r["released"]]
    good = sum(r["released_correct"] for r in released)
    proposals = [r for r in rows if r["proposal_correct"]]
    kept = sum(r["released"] and r["released_correct"] for r in proposals)
    p = good / len(released) if released else float("nan")
    rc = kept / len(proposals) if proposals else float("nan")
    f = 2 * p * rc / (p + rc) if released and proposals and p + rc else float("nan")
    return {"released": len(released), "released_correct": good, "released_wrong": len(released) - good,
            "correct_proposals": len(proposals), "correct_kept": kept,
            "repaired": sum(r["released"] and r["released_correct"] and not r["proposal_correct"] for r in rows),
            "precision": round(p, 3), "recall": round(rc, 3), "f1": round(f, 3)}


def run_cases(data: dict, series: List[dict]) -> dict:
    rows = []
    for c in data["cases"]:
        s = make_spec(c["type"], c["points"], c.get("title", "Chart"))
        truth, why = oracle(s, c["passages"], series)
        if truth != c["correct"]:
            raise SystemExit(f"oracle disagrees with the label of case {c['id']}: {why}")
        out = generation.verify_chart(s, lookup_for(data, c["passages"]), "")
        ok_out, why_out = oracle(out, c["passages"], series) if out is not None else (False, "")
        rows.append({"id": c["id"], "category": c["category"], "proposal_correct": truth, "released": out is not None,
                     "released_correct": ok_out, "released_type": out.type if out else None, "why": why_out or why})
    by_cat: Dict[str, dict] = {}
    for r in rows:
        c = by_cat.setdefault(r["category"], {"proposal_correct": r["proposal_correct"], "n": 0, "released_correct": 0,
                                              "released_wrong": 0})
        c["n"] += 1
        c["released_correct"] += r["released"] and r["released_correct"]
        c["released_wrong"] += r["released"] and not r["released_correct"]
    return {**scores(rows), "n": len(rows), "by_category": by_cat, "rows": rows}


def run_wants(data: dict) -> dict:
    rows = [{"question": w["question"], "expected": w["expected"],
             "fired": generation.wants_chart(w["question"], lookup_for(data, w["passages"]))} for w in data["wants_chart"]]
    tp = sum(r["expected"] and r["fired"] for r in rows)
    fp = sum((not r["expected"]) and r["fired"] for r in rows)
    fn = sum(r["expected"] and not r["fired"] for r in rows)
    p = tp / (tp + fp) if tp + fp else float("nan")
    rc = tp / (tp + fn) if tp + fn else float("nan")
    return {"n": len(rows), "precision": round(p, 3), "recall": round(rc, 3),
            "f1": round(2 * p * rc / (p + rc), 3) if p + rc else float("nan"), "rows": rows}


def run_e2e(data: dict, series: List[dict], model: str) -> dict:
    from langchain_ollama import ChatOllama
    kwargs = dict(model=model, temperature=0, num_ctx=4096, num_predict=300, format="json")
    if "gemma4" in model:
        kwargs["reasoning"] = False   # thinking model: otherwise the token budget goes to hidden reasoning
    llm = ChatOllama(**kwargs)
    invoke = lambda p: llm.invoke(p).content  # noqa: E731
    gen = generation.AnswerGenerator(llm_invoker=invoke, json_invoker=invoke)   # as the engine: both are set
    rows = []
    for item in data["e2e"]:
        lookup = lookup_for(data, item["passages"])
        t0 = time.time()
        proposal = gen.propose_chart(item["question"], lookup, verify=False)
        secs = round(time.time() - t0, 1)
        if proposal is None:
            rows.append({"question": item["question"], "proposed": False, "seconds": secs, "proposal_correct": False,
                         "released": False, "released_correct": False})
            continue
        truth, why = oracle(proposal, item["passages"], series)
        out = generation.verify_chart(proposal, lookup, item["question"])
        ok_out, why_out = oracle(out, item["passages"], series) if out is not None else (False, "")
        # the same proposal through the other check, so before and after are compared on identical model output
        current = generation.CHART_V2
        generation.CHART_V2 = not current
        other = generation.verify_chart(proposal, lookup, item["question"])
        generation.CHART_V2 = current
        ok_other = oracle(other, item["passages"], series)[0] if other is not None else False
        rows.append({"question": item["question"], "proposed": True, "seconds": secs, "type": proposal.type,
                     "points": [(p.label, p.value) for p in proposal.data], "proposal_correct": truth, "why": why,
                     "released": out is not None, "released_type": out.type if out else None,
                     "released_correct": ok_out, "why_released": why_out,
                     "other_check": {"released": other is not None, "released_correct": ok_other}})
    secs = sorted(r["seconds"] for r in rows)
    other_rows = [{"proposal_correct": r["proposal_correct"], **r.get("other_check", {"released": False, "released_correct": False})}
                  for r in rows]
    return {"model": model, "questions": len(rows), "proposed": sum(r["proposed"] for r in rows),
            **scores(rows), "other_check_scores": scores(other_rows), "seconds_median": secs[len(secs) // 2], "rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="*", default=["llama3"])
    ap.add_argument("--skip-e2e", action="store_true")
    ap.add_argument("--chart-v1", action="store_true", help="measure the check before chart check v2")
    args = ap.parse_args()
    logger.remove()
    if args.chart_v1:
        generation.CHART_V2 = False
    data = json.load(open(DATA_PATH, encoding="utf-8"))
    series = all_series(data)
    label = "chart check v1" if args.chart_v1 else "chart check v2"
    result = {"check": label, "cases": run_cases(data, series), "wants_chart": run_wants(data), "e2e": []}
    c = result["cases"]
    print(f"{label} | CASES ({c['n']}): released {c['released']} (correct {c['released_correct']}, wrong {c['released_wrong']}, "
          f"repaired {c['repaired']}); correct proposals kept {c['correct_kept']}/{c['correct_proposals']}; "
          f"precision={c['precision']} recall={c['recall']} F1={c['f1']}")
    for cat, v in c["by_category"].items():
        print(f"   {'correct' if v['proposal_correct'] else 'wrong  '} {cat:28} n={v['n']:<2} released correct "
              f"{v['released_correct']}, released wrong {v['released_wrong']}")
    w = result["wants_chart"]
    print(f"WANTS_CHART ({w['n']}): precision={w['precision']} recall={w['recall']} F1={w['f1']}")
    if not args.skip_e2e:
        for m in args.models:
            e = run_e2e(data, series, m)
            result["e2e"].append(e)
            print(f"E2E {m}: {e['proposed']}/{e['questions']} proposed; released {e['released']} (correct {e['released_correct']}, "
                  f"wrong {e['released_wrong']}, repaired {e['repaired']}); correct proposals kept "
                  f"{e['correct_kept']}/{e['correct_proposals']}; precision={e['precision']} recall={e['recall']} "
                  f"F1={e['f1']}; median {e['seconds_median']} s")
            o = e["other_check_scores"]
            print(f"    same proposals, {'chart check v2' if args.chart_v1 else 'chart check v1'}: released {o['released']} "
                  f"(correct {o['released_correct']}, wrong {o['released_wrong']}); correct proposals kept "
                  f"{o['correct_kept']}/{o['correct_proposals']}; precision={o['precision']} recall={o['recall']} F1={o['f1']}")
            for r in e["rows"]:
                if not r["proposed"]:
                    print(f"   no chart proposed | {r['question']}")
                    continue
                verdict = (("RELEASED " + ("correct" if r["released_correct"] else "WRONG  ")) if r["released"] else "refused         ")
                print(f"   {verdict} {r['type']}->{r['released_type'] or '-'} proposal {'ok ' if r['proposal_correct'] else 'BAD'} "
                      f"{r['points']} | {r['why_released'] or r['why']} | {r['question']}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, default=str)
    print("saved", args.out)


if __name__ == "__main__":
    main()
