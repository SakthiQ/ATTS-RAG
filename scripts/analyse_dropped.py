"""Which correct sentences does Layer 3 still drop, and why? (reviewer point 4)

Reads the independent-judge outputs (`scripts/independent_judge.py`: claims.jsonl with every draft sentence, its
evidence and the verifier's decision; judge.jsonl with the judge's label) and lists the sentences the judge calls
SUPPORTED that the verifier dropped. Each one is re-checked with the verifier to get its reason and best entailment
score, tagged with whether it carries a gold fact of the question (these are the drops that cost fact recall), and
put in one bucket:

  number guard     a number in the sentence is not in the evidence (often a reformatted number: "$1,500" vs "1500")
  contradiction    the NLI model reads the evidence as contradicting the sentence
  paraphrase       most content words are in one window, but entailment stays below the threshold
  combination      the content words are spread over several windows (the sentence joins two facts)
  low overlap      few content words appear in any window (explanations, reasons; often a lenient judge)

Usage: python scripts/analyse_dropped.py --dir eval/results/indep [--split heldout]
Writes dropped_supported.md (summary and every case) in --dir.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join("eval", "results", "indep"))
    ap.add_argument("--split", default="", help="dev or heldout (default: both)")
    args = ap.parse_args()

    from app.rag.layer3.claim_verify import DirectClaimVerifier, _terms
    from app.rag.layer3.contracts import ClaimObject

    claims = read_jsonl(os.path.join(args.dir, "claims.jsonl"))
    judge = {r["key"]: r for r in read_jsonl(os.path.join(args.dir, "judge.jsonl"))}
    facts = {}
    for path in (os.path.join("eval", "qa.jsonl"), os.path.join("eval", "qa_heldout.jsonl")):
        for q in read_jsonl(path):
            facts[q["id"]] = q.get("facts", [])
    if args.split:
        claims = [c for c in claims if c["split"] == args.split]

    labelled = [c for c in claims if c["key"] in judge]
    sup = [c for c in labelled if judge[c["key"]]["label"] == "SUPPORTED"]
    dropped = [c for c in sup if not c["kept"]]
    verifier = DirectClaimVerifier()

    rows = []
    for c in dropped:
        ev = c["evidence"]
        claim = ClaimObject(claim_id="x", order=0, text=c["claim"], evidence_ids=list(c.get("cited") or []))
        res = verifier.verify_claims([claim], ev)[0]
        terms = _terms(c["claim"])
        windows = verifier._windows(ev)
        best_one = max((len(terms & _terms(w)) / (len(terms) or 1) for _, w in windows), default=0.0)
        all_ev = set().union(*[_terms(t) for t in ev.values()]) if ev else set()
        overall = len(terms & all_ev) / (len(terms) or 1)
        reason = res.reason or ""
        if reason.startswith("Number(s)"):
            bucket = "number guard"
        elif res.status == "CONTRADICTION":
            bucket = "contradiction"
        elif best_one >= 0.6:
            bucket = "paraphrase"
        elif overall >= 0.6:
            bucket = "combination"
        else:
            bucket = "low overlap"
        low = c["claim"].lower()
        carries = any(any(a.lower() in low for a in alts) for alts in facts.get(c["qid"], []))
        rows.append(dict(split=c["split"], qid=c["qid"], claim=c["claim"], bucket=bucket, carries_fact=carries,
                         status=res.status, reason=reason[:140], best_window_overlap=round(best_one, 2),
                         evidence_overlap=round(overall, 2), judge_reason=judge[c["key"]].get("reason", "")[:160]))

    by_split = defaultdict(lambda: [0, 0, 0])
    for c in labelled:
        s = by_split[c["split"]]
        s[0] += 1
        if judge[c["key"]]["label"] == "SUPPORTED":
            s[1] += 1
            s[2] += int(not c["kept"])
    md = ["# Correct sentences Layer 3 still drops", "",
          "A sentence is *correct* when the independent judge (qwen2.5:14b, not the generator's or the verifier's "
          "family) labels it SUPPORTED by the passages. Human labels on a sample check the judge (`report.md`).", "",
          "| Split | Sentences judged | Judged supported | Supported but dropped |", "|---|---|---|---|"]
    for sp, (n, s, d) in sorted(by_split.items()):
        md.append(f"| {sp} | {n} | {s} | {d} ({100 * d / max(s, 1):.1f}% of supported) |")
    cnt = Counter(r["bucket"] for r in rows)
    fact_cnt = Counter(r["bucket"] for r in rows if r["carries_fact"])
    md += ["", "## Why they were dropped", "",
           "| Bucket | Dropped supported sentences | Of which carry a gold fact (cost recall) |", "|---|---|---|"]
    for b, n in cnt.most_common():
        md.append(f"| {b} | {n} | {fact_cnt.get(b, 0)} |")
    md += ["", "## Every case", "",
           "| Split | Question | Sentence | Bucket | Gold fact | Verifier | Overlap (best window / all evidence) |",
           "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["bucket"], r["split"], r["qid"])):
        md.append(f"| {r['split']} | {r['qid']} | {r['claim'].replace('|', '/')} | {r['bucket']} | "
                  f"{'yes' if r['carries_fact'] else ''} | {r['status']}: {r['reason'].replace('|', '/')} | "
                  f"{r['best_window_overlap']} / {r['evidence_overlap']} |")
    out = os.path.join(args.dir, "dropped_supported.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(args.dir, "dropped_supported.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\n".join(md[:20 + len(cnt)]))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
