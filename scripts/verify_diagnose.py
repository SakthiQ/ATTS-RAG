#!/usr/bin/env python3
"""Why does Layer 3 drop sentences, and could a different cut-off keep the correct ones?

Replays the cached drafts of the quality suite (no new generation), scores every claim with the verifier and
writes one row per claim: the verifier's decision, its best entailment / contradiction probability, and a
support label from the cached LLM judge and from the hand-written gold facts.

Outputs (in --out): claims.jsonl, diagnosis.md.
Usage: python scripts/verify_diagnose.py --out eval/results/verify_diag
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("scripts"))
import evaluate_pipeline as ev  # noqa: E402
from app.rag.layer3.claim_verify import _numbers, _terms  # noqa: E402

THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def judge_label(cache, sentence, reference):
    """True / False from the cached reference-based judge, None if that call is not in the cache."""
    ref = reference or "(The documents contain no information relevant to this statement.)"
    prompt = ("You are a strict fact checker. Reference text:\n" + ref + "\n\nStatement: " + sentence +
              "\n\nIs the statement fully supported by the reference text, with no extra facts the reference does not state? "
              'Reply with JSON only: {"supported": true} or {"supported": false}.')
    try:
        return bool(json.loads(cache.json(prompt)).get("supported"))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results/verify_diag")
    ap.add_argument("--model", default="llama3")
    ap.add_argument("--generator-prompt", choices=["default", "naive"], default="default")
    ap.add_argument("--cache", default=os.path.join("eval", "cache", "llm_cache.jsonl"))
    args = ap.parse_args()
    ev.quiet_logs()
    os.makedirs(args.out, exist_ok=True)

    import shutil
    import tempfile
    from app.rag.ingestion_guard import IngestionGuard

    cache = ev.LLMCache(args.cache, args.model, offline=True)  # replay only: a cache miss is skipped, never generated
    tmp = tempfile.mkdtemp(prefix="atts_diag_")
    try:
        vsm = ev.build_vsm(tmp, ev.corpus_docs(), IngestionGuard(judge=ev.make_judge_for_guard(cache)))
        system = ev.System(cache, os.path.join(args.out, "audit.jsonl"), vsm, generator_prompt=args.generator_prompt)
        gate = system.gates["l3_full"]
        ver = gate.claim_verifier
        qa = [q for q in ev.read_jsonl(os.path.join("eval", "qa.jsonl")) if q["answerable"]]

        rows, skipped = [], []
        with open(os.path.join(args.out, "claims.jsonl"), "w", encoding="utf-8") as sink:
            for q in qa:
                items, stage = system.retrieve(q["question"], use_l2=True)
                try:
                    draft = gate.generator.generate(q["question"], items)
                except RuntimeError:
                    skipped.append(q["id"])
                    continue
                if not draft.claims:
                    continue
                lookup = draft.evidence_lookup
                windows = ver._windows(lookup)
                all_nums = set().union(*[_numbers(t) for t in lookup.values()]) if lookup else set()
                decided = {r.claim_id: r for r in ver.verify_claims(draft.claims, lookup)}
                for claim in draft.claims:
                    cands = ver._candidates(claim, windows)
                    missing = sorted(_numbers(claim.text) - all_nums)
                    extractive = ver._extractive_match(claim, cands) is not None
                    best_ent = best_con = 0.0
                    best_overlap = 0.0
                    if cands:
                        probs = [ver._softmax(r) for r in ver._nli_model.predict([(p, claim.text) for _, p in cands], batch_size=32)]
                        ent = [float(p[ver._i_entail]) for p in probs]
                        con = [float(p[ver._i_contra]) for p in probs]
                        i = max(range(len(ent)), key=ent.__getitem__)
                        best_ent, best_con = ent[i], max(con)
                        ct = _terms(claim.text)
                        best_overlap = len(ct & _terms(cands[i][1])) / (len(ct) or 1)
                    res = decided[claim.claim_id]
                    low = claim.text.lower()
                    has_gold = any(any(a.lower() in low for a in alts) for alts in q["facts"])
                    row = dict(qid=q["id"], claim=claim.text, cited=claim.evidence_ids, status=res.status, kept=res.status == "ENTAILMENT",
                               best_entail=round(best_ent, 4), best_contra=round(best_con, 4), window_overlap=round(best_overlap, 3),
                               missing_numbers=missing, extractive=extractive, has_gold_fact=has_gold,
                               n_words=len(claim.text.split()), judge_supported=judge_label(cache, claim.text, q["reference"]))
                    rows.append(row)
                    sink.write(json.dumps(row) + "\n")
                print(f"[{q['id']}] {len(draft.claims)} claims", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ------------------------------------------------------------------ analysis
    labelled = [r for r in rows if r["judge_supported"] is not None]
    sup = [r for r in labelled if r["judge_supported"]]
    uns = [r for r in labelled if not r["judge_supported"]]
    md = ["# Layer 3 verification diagnosis", "",
          f"{len(rows)} claims from {len(set(r['qid'] for r in rows))} questions (skipped, not in cache: {skipped or 'none'}). "
          f"{len(labelled)} carry a cached judge label (the judge is the same llama3 model, so labels are noisy).", "",
          f"Kept by the verifier: {sum(r['kept'] for r in rows)} of {len(rows)} ({100 * sum(r['kept'] for r in rows) / max(len(rows), 1):.1f}%).", ""]

    md += ["## Judge-supported claims: how many does the verifier keep?", ""]
    kept_sup = sum(r["kept"] for r in sup)
    md.append(f"- supported by the judge: {len(sup)}; kept {kept_sup} ({100 * kept_sup / max(len(sup), 1):.1f}%), wrongly dropped {len(sup) - kept_sup}")
    kept_uns = sum(r["kept"] for r in uns)
    md.append(f"- unsupported by the judge: {len(uns)}; wrongly kept {kept_uns} ({100 * kept_uns / max(len(uns), 1):.1f}%)")
    gold = [r for r in rows if r["has_gold_fact"]]
    md.append(f"- claims that state a gold fact: {len(gold)}; kept {sum(r['kept'] for r in gold)} ({100 * sum(r['kept'] for r in gold) / max(len(gold), 1):.1f}%)")
    md.append("")

    def cause(r):
        if r["missing_numbers"]:
            return "number not found in evidence"
        if r["status"] == "CONTRADICTION":
            return "NLI says contradiction"
        if r["best_entail"] >= 0.30:
            return "borderline entailment (0.30-0.60)"
        if r["window_overlap"] < 0.4:
            return "low word overlap with every window (synthesis / paraphrase)"
        return "entailment below 0.30 despite overlap"

    dropped_ok = [r for r in sup if not r["kept"]]
    md += ["## Why are judge-supported claims dropped?", "", "| Cause | Claims |", "| --- | --- |"]
    for c, n in Counter(cause(r) for r in dropped_ok).most_common():
        md.append(f"| {c} | {n} |")
    md += ["", "Examples (first 12):", ""]
    for r in dropped_ok[:12]:
        md.append(f"- ({cause(r)}; entail {r['best_entail']:.2f}) {r['claim']}")

    md += ["", "## Cut-off sweep (claims decided by NLI only; number guard and extractive match applied first, as in production)", "",
           "| Entailment cut-off | Supported claims kept | Unsupported claims kept |", "| --- | --- | --- |"]

    def kept_at(r, t):
        if r["missing_numbers"]:
            return False
        if r["extractive"]:
            return True
        return r["best_entail"] >= t

    for t in THRESHOLDS:
        a = sum(kept_at(r, t) for r in sup)
        b = sum(kept_at(r, t) for r in uns)
        md.append(f"| {t:.2f} | {a}/{len(sup)} ({100 * a / max(len(sup), 1):.0f}%) | {b}/{len(uns)} ({100 * b / max(len(uns), 1):.0f}%) |")
    md += ["", "Rows with a higher cut-off keep fewer claims of both kinds; the current production cut-off is 0.60."]
    with open(os.path.join(args.out, "diagnosis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
