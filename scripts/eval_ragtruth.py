"""Layer 3 verifier on RAGTruth (public benchmark, reviewer points 1 and 4).

RAGTruth (Niu et al., ACL 2024; https://github.com/ParticleMedia/RAGTruth) has 2,700 test answers written by six
models (GPT-4, GPT-3.5, Mistral-7B, Llama-2-7B/13B/70B) for three tasks (question answering, data-to-text, news
summarisation), with every hallucinated span marked by human annotators. We run the Layer 3 claim verifier exactly as
the pipeline does (`DirectClaimVerifier`, default thresholds, no second opinion) on each answer: the answer is split
into sentences, each sentence is checked against the source, and a sentence is *flagged* unless the verifier finds it
ENTAILED. Nothing is tuned on RAGTruth.

Protocol, following the official code (`baseline/prepare_dataset.py`, `predict_and_evaluate.py`): test split, answers
with quality "good" (2,675), an answer is hallucinated when it has at least one labelled span.
- Response level (comparable with the paper's Table 5): an answer is predicted hallucinated when any sentence is flagged.
- Sentence level (ours; the paper scores character spans, so these are not comparable with its Table 6): a sentence
  is hallucinated when a labelled span overlaps it.

Usage:
  python scripts/eval_ragtruth.py --data D:/ATTS-RAG-data/benchmarks/ragtruth/dataset --out eval/results/ragtruth
  (resumable: answers already in items.jsonl are skipped; --report-only rebuilds report.md)
"""
import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Published response-level results, RAGTruth paper Table 5 (precision, recall, F1 in %), per task and overall.
PAPER_TABLE5 = {
    "Prompt (gpt-3.5-turbo)":       {"QA": (18.8, 84.4, 30.8), "Data2txt": (65.1, 95.5, 77.4), "Summary": (23.4, 89.2, 37.1), "Overall": (37.1, 92.3, 52.9)},
    "Prompt (gpt-4-turbo)":         {"QA": (33.2, 90.6, 45.6), "Data2txt": (64.3, 100.0, 78.3), "Summary": (31.5, 97.6, 47.6), "Overall": (46.9, 97.9, 63.4)},
    "SelfCheckGPT (gpt-3.5-turbo)": {"QA": (35.0, 58.0, 43.7), "Data2txt": (68.2, 82.8, 74.8), "Summary": (31.1, 56.5, 40.1), "Overall": (49.7, 71.9, 58.8)},
    "LMvLM (gpt-4-turbo)":          {"QA": (18.7, 76.9, 30.1), "Data2txt": (68.0, 76.7, 72.1), "Summary": (23.3, 81.9, 36.2), "Overall": (36.2, 77.8, 49.4)},
    "Fine-tuned Llama-2-13B":       {"QA": (61.6, 76.3, 68.2), "Data2txt": (85.4, 91.0, 88.1), "Summary": (64.0, 54.9, 59.1), "Overall": (76.9, 80.7, 78.7)},
}
TASKS = ["QA", "Data2txt", "Summary"]


# ------------------------------------------------------------------------------------------ evidence
def _flat(prefix: str, value: Any, out: List[str]) -> None:
    if value is None:
        return
    if isinstance(value, dict):
        for k, v in value.items():
            _flat(f"{prefix} {k}".strip(), v, out)
    elif isinstance(value, list):
        out.append(f"{prefix}: {', '.join(str(v) for v in value)}")
    else:
        if isinstance(value, bool):
            value = "yes" if value else "no"
        out.append(f"{prefix}: {value}.")


def evidence_for(src: Dict[str, Any]) -> Dict[str, str]:
    """The source as passages: QA passages, the news article, or the business record (facts, then one passage per review)."""
    info, task = src["source_info"], src["task_type"]
    if task == "QA":
        parts = [p.strip() for p in re.split(r"passage \d+:", info["passages"]) if p.strip()]
        ev = {f"p{i + 1}": p for i, p in enumerate(parts)}
        return ev or {"p1": info["passages"]}
    if task == "Summary":
        return {"article": info if isinstance(info, str) else json.dumps(info)}
    # Data2txt: Yelp business record
    info = dict(info)
    reviews = info.pop("review_info", []) or []
    lines: List[str] = []
    for k, v in info.items():
        _flat(k.replace("_", " "), v, lines)
    ev = {"record": "\n".join(lines)}
    for i, r in enumerate(reviews):
        stars, date, text = r.get("review_stars"), r.get("review_date"), r.get("review_text", "")
        ev[f"review{i + 1}"] = f"Review ({stars} stars, {date}): {text}"
    return ev


def sentence_spans(text: str, split_sentences) -> List[Tuple[int, int, str]]:
    """Sentences of the answer with their character offsets (the labels are character spans)."""
    out, cursor = [], 0
    for s in split_sentences(text):
        probe = s.strip()[:40]
        at = text.find(probe, cursor) if probe else -1
        if at < 0:
            at = cursor
        end = at + len(s.strip())
        out.append((at, min(end, len(text)), s))
        cursor = max(cursor, end)
    return out


# ------------------------------------------------------------------------------------------ metrics
def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return 100 * p, 100 * r, 100 * f


def boot_f1(gold: List[int], pred: List[int], n: int = 2000, seed: int = 7) -> Tuple[float, float]:
    rng = random.Random(seed)
    idx = list(range(len(gold)))
    vals = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        tp = sum(1 for i in s if gold[i] and pred[i])
        fp = sum(1 for i in s if not gold[i] and pred[i])
        fn = sum(1 for i in s if gold[i] and not pred[i])
        vals.append(prf(tp, fp, fn)[2])
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


def counts(gold: List[int], pred: List[int]) -> Tuple[int, int, int]:
    tp = sum(1 for g, p in zip(gold, pred) if g and p)
    fp = sum(1 for g, p in zip(gold, pred) if not g and p)
    fn = sum(1 for g, p in zip(gold, pred) if g and not p)
    return tp, fp, fn


def md_table(head: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def report(items: List[Dict[str, Any]], out_dir: str) -> str:
    md = ["## Layer 3 verifier on RAGTruth (test split, quality = good)", "",
          f"{len(items)} answers. An answer is *predicted hallucinated* when the verifier leaves any of its sentences "
          "unsupported. Precision, recall and F1 in %, with a bootstrap 95% interval for our F1. The published rows are "
          "RAGTruth Table 5 (same split and label rule). Nothing was tuned on RAGTruth.", ""]
    rows = []
    for task in TASKS + ["Overall"]:
        sel = [it for it in items if task == "Overall" or it["task"] == task]
        if not sel:
            continue
        g = [int(it["gold"]) for it in sel]
        p = [int(it["pred"]) for it in sel]
        P, R, F = prf(*counts(g, p))
        lo, hi = boot_f1(g, p)
        rows.append([f"**ATTS-RAG Layer 3 verifier** ({task})", f"{P:.1f}", f"{R:.1f}", f"{F:.1f} [{lo:.0f}, {hi:.0f}]",
                     f"{sum(g)}/{len(g)}"])
    md += [md_table(["Method (task)", "Precision", "Recall", "F1", "Hallucinated answers"], rows), ""]
    prow = []
    for name, d in PAPER_TABLE5.items():
        prow.append([name] + [f"{d[t][2]:.1f}" for t in TASKS + ["Overall"]])
    ours = []
    for task in TASKS + ["Overall"]:
        sel = [it for it in items if task == "Overall" or it["task"] == task]
        ours.append(f"{prf(*counts([int(i['gold']) for i in sel], [int(i['pred']) for i in sel]))[2]:.1f}" if sel else "-")
    prow.append(["**ATTS-RAG Layer 3 verifier (ours, no training on RAGTruth)**"] + ours)
    md += ["### Response-level F1 against the published baselines", "",
           md_table(["Method", "QA", "Data-to-text", "Summary", "Overall"], prow), "",
           "Note: in the paper's Table 5, the GPT-4 prompt QA cell gives P 33.2 and R 90.6, which imply F1 48.6, not "
           "45.6 as printed; the printed value is kept.", ""]

    # sentence level
    srows = []
    for task in TASKS + ["Overall"]:
        sel = [s for it in items if task == "Overall" or it["task"] == task for s in it["sentences"]]
        if not sel:
            continue
        g = [int(s["gold"]) for s in sel]
        p = [int(not s["kept"]) for s in sel]
        P, R, F = prf(*counts(g, p))
        clean = [s for s in sel if not s["gold"]]
        kept_clean = sum(s["kept"] for s in clean)
        srows.append([task, len(sel), f"{P:.1f}", f"{R:.1f}", f"{F:.1f}",
                      f"{100 * kept_clean / max(len(clean), 1):.1f}% ({kept_clean}/{len(clean)})"])
    md += ["### Sentence level (ours; not comparable with the paper's character-span Table 6)", "",
           "A sentence is hallucinated when a labelled span overlaps it. *Clean sentences kept* = sentences with no "
           "labelled span that the verifier released (the cost of verification, reviewer point 4).", "",
           md_table(["Task", "Sentences", "Precision", "Recall", "F1", "Clean sentences kept"], srows), ""]

    # by generator model
    mrows = []
    for m in sorted({it["model"] for it in items}):
        sel = [it for it in items if it["model"] == m]
        g = [int(it["gold"]) for it in sel]
        p = [int(it["pred"]) for it in sel]
        P, R, F = prf(*counts(g, p))
        mrows.append([m, len(sel), f"{100 * sum(g) / len(g):.1f}%", f"{P:.1f}", f"{R:.1f}", f"{F:.1f}"])
    md += ["### By the model that wrote the answer (response level, all tasks)", "",
           md_table(["Generator", "Answers", "Hallucinated", "Precision", "Recall", "F1"], mrows), ""]

    # why clean sentences are dropped
    reasons = defaultdict(int)
    for it in items:
        for s in it["sentences"]:
            if not s["gold"] and not s["kept"]:
                r = s.get("reason") or ""
                key = "number not in source" if r.startswith("Number(s)") else (s["status"] or "?")
                reasons[(it["task"], key)] += 1
    rrows = [[t, k, n] for (t, k), n in sorted(reasons.items(), key=lambda x: (-x[1]))]
    md += ["### Why clean sentences were flagged", "", md_table(["Task", "Verifier outcome", "Sentences"], rrows), ""]
    text = "\n".join(md)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return text


# ------------------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="D:/ATTS-RAG-data/benchmarks/ragtruth/dataset")
    ap.add_argument("--out", default=os.path.join("eval", "results", "ragtruth"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--tasks", default="QA,Data2txt,Summary")
    ap.add_argument("--limit", type=int, default=0, help="first N answers per task (quick run)")
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads (keep low when other jobs run)")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "items.jsonl")
    done: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            done[r["id"]] = r

    if not args.report_only:
        import torch
        torch.set_num_threads(args.threads)
        from app.rag.layer3.claim_verify import DirectClaimVerifier, split_sentences
        from app.rag.layer3.contracts import ClaimObject

        src = {}
        for line in open(os.path.join(args.data, "source_info.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            src[r["source_id"]] = r
        tasks = set(args.tasks.split(","))
        resp = [json.loads(l) for l in open(os.path.join(args.data, "response.jsonl"), encoding="utf-8")]
        resp = [r for r in resp if r["split"] == args.split and r["quality"] == "good" and src[r["source_id"]]["task_type"] in tasks]
        if args.limit:
            per = defaultdict(int)
            keep = []
            for r in resp:
                t = src[r["source_id"]]["task_type"]
                if per[t] < args.limit:
                    keep.append(r)
                    per[t] += 1
            resp = keep
        todo = [r for r in resp if r["id"] not in done]
        print(f"{len(resp)} answers, {len(todo)} to do", flush=True)
        verifier = DirectClaimVerifier()
        with open(path, "a", encoding="utf-8") as sink:
            for n, r in enumerate(todo, 1):
                s = src[r["source_id"]]
                ev = evidence_for(s)
                spans = sentence_spans(r["response"], split_sentences)
                claims = [ClaimObject(claim_id=f"c{i}", order=i, text=t.strip(), evidence_ids=list(ev))
                          for i, (_, _, t) in enumerate(spans) if t.strip()]
                results = verifier.verify_claims(claims, ev) if claims else []
                by_id = {x.claim_id: x for x in results}
                sents = []
                for i, (a, b, t) in enumerate(spans):
                    if not t.strip():
                        continue
                    res = by_id.get(f"c{i}")
                    gold = any(lab["start"] < b and lab["end"] > a for lab in r["labels"])
                    sents.append(dict(text=t.strip(), start=a, end=b, gold=gold,
                                      status=res.status if res else None, kept=bool(res and res.status == "ENTAILMENT"),
                                      reason=(res.reason or "")[:160] if res else ""))
                item = dict(id=r["id"], task=s["task_type"], model=r["model"], gold=bool(r["labels"]),
                            pred=any(not x["kept"] for x in sents), n_labels=len(r["labels"]), sentences=sents)
                sink.write(json.dumps(item) + "\n")
                sink.flush()
                done[r["id"]] = item
                if n % 50 == 0:
                    print(f"[ragtruth {n}/{len(todo)}]", flush=True)
    print(report(list(done.values()), args.out))


if __name__ == "__main__":
    main()
