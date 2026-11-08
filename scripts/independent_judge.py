#!/usr/bin/env python3
"""Independent check of Layer 3: a local judge from another model family, validated against human labels.

The generator is Llama 3 and the verifier is a small NLI model; the judge is a different, larger open model served by
Ollama (default qwen2.5:14b), so its errors are not the generator's or the verifier's. It costs nothing to run. Its
labels are trusted only as far as they agree with a human: the `sheet` step writes a blind sample for a person to label,
and `report` gives the agreement (Cohen's kappa) of judge and verifier with those labels.

Steps (each reads the previous step's files in --out):
  claims  replay the quality drafts (eval/qa.jsonl and, with --heldout, eval/qa_heldout.jsonl), run the full Layer 3
          verifier and write claims.jsonl: question, sentence, the evidence passages, the verifier's decision.
          --offline uses only cached drafts (a missing one is skipped).
  judge   label every sentence SUPPORTED / NOT_SUPPORTED / CONTRADICTED against the evidence passages (judge.jsonl;
          resumable: sentences already judged are skipped).
  sheet   write label_sheet.csv: a blind, stratified sample (verifier kept / dropped) for a human to label. The
          verifier's and the judge's labels are not in the sheet. Fill the `human_label` column with S, N or C.
  report  report.md: verifier vs judge on all sentences, and judge / verifier vs the human labels.

Usage:
  python scripts/independent_judge.py claims --out eval/results/indep --heldout
  python scripts/independent_judge.py judge  --out eval/results/indep --host http://127.0.0.1:11435
  python scripts/independent_judge.py sheet  --out eval/results/indep --n 100
  python scripts/independent_judge.py report --out eval/results/indep
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
import urllib.request
from collections import Counter

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("scripts"))

LABELS = ("SUPPORTED", "NOT_SUPPORTED", "CONTRADICTED")
SHORT = {"S": "SUPPORTED", "N": "NOT_SUPPORTED", "C": "CONTRADICTED"}
JUDGE_PROMPT = (
    "You check whether a statement is supported by the evidence passages below, and by nothing else: ignore what you "
    "know about the world.\n\n{evidence}\n\nStatement: {claim}\n\n"
    "Answer SUPPORTED only if everything the statement asserts (names, numbers, dates, conditions) is stated in the "
    "passages, possibly in other words; CONTRADICTED if a passage states something incompatible with it; NOT_SUPPORTED "
    "otherwise. Reasons, purposes, consequences, exclusions or generalisations that the passages do not state are "
    "NOT_SUPPORTED even when they are plausible (e.g. 'this rule exists to protect accounts' when the passage only gives "
    "the rule). "
    'Reply with JSON: {{"label": "SUPPORTED" | "NOT_SUPPORTED" | "CONTRADICTED", "reason": "<one short sentence>"}}'
)
SCHEMA = {"type": "object", "properties": {"label": {"type": "string", "enum": list(LABELS)}, "reason": {"type": "string"}},
          "required": ["label", "reason"]}


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.exists(path) else []


def key_of(row):
    return hashlib.sha1((row["qid"] + "\n" + row["claim"]).encode("utf-8")).hexdigest()[:16]


def evidence_text(row):
    return "\n\n".join(f"[{eid}] {txt}" for eid, txt in row["evidence"].items())


# ---------------------------------------------------------------------------------------------------- claims
def step_claims(args):
    import shutil
    import tempfile
    import evaluate_pipeline as ev
    from app.rag.ingestion_guard import IngestionGuard
    ev.quiet_logs()
    ev.EXTRA_CORPUS = os.path.join("eval", "corpus_heldout") if args.heldout else ""
    cache = ev.LLMCache(args.cache, args.model, offline=args.offline)
    tmp = tempfile.mkdtemp(prefix="atts_indep_")
    rows, skipped = [], []
    try:
        vsm = ev.build_vsm(tmp, ev.corpus_docs(), IngestionGuard(judge=ev.make_judge_for_guard(cache)))
        system = ev.System(cache, os.path.join(args.out, "audit.jsonl"), vsm)
        gate = system.gates["l3_full"]
        files = [("dev", os.path.join("eval", "qa.jsonl"))] + ([("heldout", os.path.join("eval", "qa_heldout.jsonl"))] if args.heldout else [])
        for split, path in files:
            for q in [q for q in ev.read_jsonl(path) if q["answerable"]]:
                items, _ = system.retrieve(q["question"], use_l2=True)
                if not items:
                    continue
                try:
                    draft = gate.generator.generate(q["question"], items)
                except RuntimeError:
                    skipped.append(q["id"])
                    continue
                decided = {r.claim_id: r for r in gate.claim_verifier.verify_claims(draft.claims, draft.evidence_lookup)}
                for c in draft.claims:
                    r = decided[c.claim_id]
                    rows.append(dict(split=split, qid=q["id"], question=q["question"], claim=c.text, cited=c.evidence_ids,
                                     evidence=draft.evidence_lookup, verifier=r.status, kept=r.status == "ENTAILMENT"))
                print(f"[{split} {q['id']}] {len(draft.claims)} sentences", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    with open(os.path.join(args.out, "claims.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(dict(r, key=key_of(r)), ensure_ascii=False) + "\n")
    print(f"{len(rows)} sentences; skipped (draft not cached): {skipped or 'none'}")


# ---------------------------------------------------------------------------------------------------- judge
def ask_judge(host, model, prompt, timeout=600):
    body = json.dumps({"model": model, "stream": False, "format": SCHEMA, "options": {"temperature": 0, "num_ctx": 8192},
                       "messages": [{"role": "user", "content": prompt}]}).encode("utf-8")
    req = urllib.request.Request(host.rstrip("/") + "/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    data = json.loads(out["message"]["content"])
    if data.get("label") not in LABELS:
        raise ValueError(f"bad label: {data}")
    return data


def step_judge(args):
    claims = read_jsonl(os.path.join(args.out, "claims.jsonl"))
    path = os.path.join(args.out, "judge.jsonl")
    done = {r["key"] for r in read_jsonl(path)}
    todo = [r for r in claims if r["key"] not in done]
    print(f"{len(claims)} sentences, {len(done)} already judged, {len(todo)} to go ({args.judge_model} at {args.host})", flush=True)
    with open(path, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            prompt = JUDGE_PROMPT.format(evidence=evidence_text(r), claim=r["claim"])
            try:
                v = ask_judge(args.host, args.judge_model, prompt)
            except Exception as exc:
                print(f"  {r['key']}: judge failed ({exc}); rerun to retry", flush=True)
                continue
            f.write(json.dumps(dict(key=r["key"], label=v["label"], reason=v.get("reason", ""), judge=args.judge_model)) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"  {i}/{len(todo)}", flush=True)


# ---------------------------------------------------------------------------------------------------- sheet
def step_sheet(args):
    claims = read_jsonl(os.path.join(args.out, "claims.jsonl"))
    rng = random.Random(args.seed)
    kept = [r for r in claims if r["kept"]]
    dropped = [r for r in claims if not r["kept"]]
    half = args.n // 2
    sample = rng.sample(kept, min(half, len(kept))) + rng.sample(dropped, min(args.n - half, len(dropped)))
    rng.shuffle(sample)
    path = os.path.join(args.out, "label_sheet.csv")
    if os.path.exists(path) and not args.force:
        raise SystemExit(f"{path} exists (it may hold your labels); pass --force to overwrite")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "question", "statement", "evidence", "human_label (S / N / C)", "note"])
        for r in sample:
            w.writerow([r["key"], r["question"], r["claim"], evidence_text(r), "", ""])
    print(f"wrote {path}: {len(sample)} sentences ({min(half, len(kept))} the verifier kept, the rest it dropped; shuffled, labels hidden)")


# ---------------------------------------------------------------------------------------------------- report
def kappa(a, b):
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def binary(label):
    return "SUPPORTED" if label == "SUPPORTED" else "NOT"


def step_report(args):
    import evaluate_pipeline as ev
    claims = {r["key"]: r for r in read_jsonl(os.path.join(args.out, "claims.jsonl"))}
    judge = {r["key"]: r for r in read_jsonl(os.path.join(args.out, "judge.jsonl"))}
    human = {}
    sheet = os.path.join(args.out, "label_sheet.csv")
    if os.path.exists(sheet):
        for row in csv.DictReader(open(sheet, encoding="utf-8-sig")):
            lab = (row.get("human_label (S / N / C)") or "").strip().upper()[:1]
            if lab in SHORT:
                human[row["key"]] = SHORT[lab]
    verdict = lambda k: "SUPPORTED" if claims[k]["kept"] else "NOT"  # noqa: E731

    md = ["# Independent check of Layer 3", "",
          f"{len(claims)} draft sentences · judge: {next(iter(judge.values()))['judge'] if judge else '-'} (local, another model family) · "
          f"human labels: {len(human)}", ""]

    def verifier_vs(labels, who):
        keys = [k for k in labels if k in claims]
        sup = [k for k in keys if labels[k] == "SUPPORTED"]
        uns = [k for k in keys if labels[k] != "SUPPORTED"]
        kept_sup = sum(claims[k]["kept"] for k in sup)
        kept_uns = sum(claims[k]["kept"] for k in uns)
        kept = kept_sup + kept_uns
        return [f"### Verifier against the {who} ({len(keys)} sentences)", "",
                ev.md_table(["Measure", "Value"], [
                    ["Supported sentences the verifier kept (recall)", ev.fmt_prop(kept_sup, len(sup))],
                    ["Unsupported sentences the verifier released", ev.fmt_prop(kept_uns, len(uns))],
                    ["Released sentences that are supported (precision)", ev.fmt_prop(kept_sup, kept)],
                    ["Agreement (supported vs not)", ev.fmt_prop(sum(verdict(k) == binary(labels[k]) for k in keys), len(keys))],
                    ["Cohen's kappa (supported vs not)", f"{kappa([verdict(k) for k in keys], [binary(labels[k]) for k in keys]):.2f}"]]), ""]

    if human:
        both = [k for k in human if k in judge]
        md += ["## Is the judge trustworthy? Judge against the human labels", "",
               ev.md_table(["Measure", "Value"], [
                   ["Sentences labelled by both", str(len(both))],
                   ["Agreement, three labels", ev.fmt_prop(sum(judge[k]["label"] == human[k] for k in both), len(both))],
                   ["Cohen's kappa, three labels", f"{kappa([judge[k]['label'] for k in both], [human[k] for k in both]):.2f}"],
                   ["Agreement, supported vs not", ev.fmt_prop(sum(binary(judge[k]['label']) == binary(human[k]) for k in both), len(both))],
                   ["Cohen's kappa, supported vs not", f"{kappa([binary(judge[k]['label']) for k in both], [binary(human[k]) for k in both]):.2f}"]]), "",
               "Human labels:", "", ev.md_table(["Label", "Sentences"], [[l, c] for l, c in Counter(human.values()).most_common()]), ""]
        md += verifier_vs(human, "human labels")
    if judge:
        md += verifier_vs({k: v["label"] for k, v in judge.items()}, "judge (all sentences)")
        for split in sorted({r["split"] for r in claims.values()}):
            md += verifier_vs({k: v["label"] for k, v in judge.items() if claims.get(k, {}).get("split") == split}, f"judge, {split} questions")
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["claims", "judge", "sheet", "report"])
    ap.add_argument("--out", default=os.path.join("eval", "results", "indep"))
    ap.add_argument("--heldout", action="store_true", help="claims: also the held-out questions (eval/qa_heldout.jsonl)")
    ap.add_argument("--offline", action="store_true", help="claims: only cached drafts")
    ap.add_argument("--model", default="llama3", help="claims: the generator")
    ap.add_argument("--cache", default=os.path.join("eval", "cache", "llm_cache.jsonl"))
    ap.add_argument("--host", default=os.getenv("JUDGE_OLLAMA_HOST", "http://127.0.0.1:11435"), help="judge: Ollama server of the judge")
    ap.add_argument("--judge-model", default="qwen2.5:14b")
    ap.add_argument("--n", type=int, default=100, help="sheet: sentences to label")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--force", action="store_true", help="sheet: overwrite an existing sheet")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    {"claims": step_claims, "judge": step_judge, "sheet": step_sheet, "report": step_report}[args.step](args)


if __name__ == "__main__":
    main()
