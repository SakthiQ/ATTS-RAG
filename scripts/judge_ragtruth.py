"""Validate the independent judge against human labels: RAGTruth sentences (feedback point 5).

The independent evaluator (`scripts/independent_judge.py`, qwen2.5:14b) labels each answer sentence SUPPORTED /
NOT_SUPPORTED / CONTRADICTED. Before its labels are trusted on our own answers, its agreement with HUMAN labels is
measured on a public set: RAGTruth (human annotators marked every hallucinated span of 2,700 test answers).

A stratified sample of RAGTruth test sentences (QA and summarisation; data-to-text records are long and unlike our
documents) is drawn with a fixed seed: half overlap a human-labelled span ("hallucinated"), half do not ("clean").
The judge sees the same source passages and the same prompt it uses on ATTS-RAG answers. Agreement is reported as
accuracy and Cohen's kappa on the binary decision (judge SUPPORTED vs not; human clean vs hallucinated), with the
Layer 3 verifier on the same sentences for comparison.

Needs eval/results/ragtruth/items.jsonl from scripts/eval_ragtruth.py.
Usage: python scripts/judge_ragtruth.py --n 300 --out eval/results/judge_ragtruth   (resumable)
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("scripts"))


def kappa(a, b):
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def boot_kappa(a, b, n=2000, seed=3):
    rng = random.Random(seed)
    idx = list(range(len(a)))
    vals = [kappa([a[i] for i in s], [b[i] for i in s]) for s in ([rng.choice(idx) for _ in idx] for _ in range(n))]
    vals = sorted(v for v in vals if v == v)   # drop resamples with a single class (kappa undefined)
    if not vals:
        return float("nan"), float("nan")
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="D:/ATTS-RAG-data/benchmarks/ragtruth/dataset")
    ap.add_argument("--items", default=os.path.join("eval", "results", "ragtruth", "items.jsonl"))
    ap.add_argument("--out", default=os.path.join("eval", "results", "judge_ragtruth"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--tasks", default="QA,Summary")
    ap.add_argument("--host", default=os.getenv("JUDGE_OLLAMA_HOST", "http://127.0.0.1:11435"))
    ap.add_argument("--judge-model", default="qwen2.5:14b")
    args = ap.parse_args()

    import evaluate_pipeline as ev
    from eval_ragtruth import evidence_for
    from independent_judge import JUDGE_PROMPT, ask_judge

    os.makedirs(args.out, exist_ok=True)
    src = {}
    for line in open(os.path.join(args.data, "source_info.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        src[r["source_id"]] = r
    resp = {}
    for line in open(os.path.join(args.data, "response.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        resp[r["id"]] = r
    tasks = set(args.tasks.split(","))
    pool_h, pool_c = [], []
    for line in open(args.items, encoding="utf-8"):
        it = json.loads(line)
        if it["task"] not in tasks:
            continue
        for j, s in enumerate(it["sentences"]):
            if len(s["text"].split()) < 4:
                continue
            row = dict(key=f"{it['id']}#{j}", id=it["id"], task=it["task"], model=it["model"], sentence=s["text"],
                       human_hallucinated=bool(s["gold"]), verifier_kept=bool(s["kept"]))
            (pool_h if s["gold"] else pool_c).append(row)
    rng = random.Random(args.seed)
    rng.shuffle(pool_h)
    rng.shuffle(pool_c)
    half = args.n // 2
    sample = pool_h[:half] + pool_c[:args.n - min(half, len(pool_h))]
    rng.shuffle(sample)

    path = os.path.join(args.out, "judge.jsonl")
    done = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            done[r["key"]] = r
    todo = [r for r in sample if r["key"] not in done]
    print(f"{len(sample)} sentences ({sum(r['human_hallucinated'] for r in sample)} hallucinated), {len(todo)} to judge", flush=True)
    with open(path, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            evid = evidence_for(src[resp[r["id"]]["source_id"]])
            evidence = "\n\n".join(f"[{k}] {v}" for k, v in evid.items())
            try:
                v = ask_judge(args.host, args.judge_model, JUDGE_PROMPT.format(evidence=evidence, claim=r["sentence"]))
            except Exception as exc:
                print(f"  {r['key']}: judge failed ({exc}); rerun to retry", flush=True)
                continue
            out = dict(r, judge=v["label"], judge_reason=v.get("reason", ""))
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            done[r["key"]] = out
            if i % 20 == 0:
                print(f"  {i}/{len(todo)}", flush=True)

    rows = [done[r["key"]] for r in sample if r["key"] in done]
    human = [int(r["human_hallucinated"]) for r in rows]
    judge = [int(r["judge"] != "SUPPORTED") for r in rows]
    verif = [int(not r["verifier_kept"]) for r in rows]

    def line(name, pred, sel=None):
        idx = [i for i in range(len(rows)) if sel is None or rows[i]["task"] == sel]
        h = [human[i] for i in idx]
        p = [pred[i] for i in idx]
        if not idx:
            return None
        acc = sum(x == y for x, y in zip(h, p)) / len(h)
        k = kappa(h, p)
        lo, hi = boot_kappa(h, p)
        tp = sum(1 for x, y in zip(h, p) if x and y)
        caught = f"{tp}/{sum(h)}"
        false_flag = f"{sum(1 for x, y in zip(h, p) if not x and y)}/{len(h) - sum(h)}"
        return [name + (f" ({sel})" if sel else ""), len(h), f"{100 * acc:.1f}%", f"{k:.2f} [{lo:.2f}, {hi:.2f}]", caught, false_flag]

    table = []
    for sel in [None] + sorted(tasks):
        for name, pred in ((f"Judge ({args.judge_model})", judge), ("Layer 3 verifier", verif)):
            ln = line(name, pred, sel)
            if ln:
                table.append(ln)
    md = ["## Independent judge vs human labels (RAGTruth)", "",
          f"{len(rows)} RAGTruth test sentences (QA and summarisation), stratified: {sum(human)} overlap a human-labelled "
          "hallucinated span, the rest do not. Binary decision: SUPPORTED vs not (judge), kept vs dropped (verifier). "
          "Cohen's kappa with a bootstrap 95% interval. Same judge prompt as on ATTS-RAG answers. Because the sample is "
          "balanced, accuracy and kappa are not the rates on natural answers, where hallucinated sentences are rarer.", "",
          ev.md_table(["Rater", "Sentences", "Agreement with humans", "Cohen's kappa", "Hallucinated sentences caught",
                       "Clean sentences flagged"], table), "",
          f"Judge labels: { {l: sum(1 for r in rows if r['judge'] == l) for l in ('SUPPORTED', 'NOT_SUPPORTED', 'CONTRADICTED')} }", ""]
    text = "\n".join(md)
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
