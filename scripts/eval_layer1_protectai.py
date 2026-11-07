#!/usr/bin/env python3
"""Baseline for Layer 1: an existing, published prompt-injection detector on the same rows as eval_layer1_public.py.

Model: protectai/deberta-v3-base-prompt-injection-v2 (Apache-2.0), a DeBERTa-v3-base classifier with labels SAFE /
INJECTION, used at its default decision (INJECTION probability >= 0.5). Two ways of running it:
  whole    the model's standard use: the whole message, truncated to 512 tokens;
  segment  the same model applied to the gate's segments (app/rag/threat_gate.segment_text), INJECTION if any segment
           is - so the comparison with ATTS-RAG's segment scoring is like for like.
The detector only says attack / not attack, so its result is reported as "blocked" (it has no flag level).

Usage: python scripts/eval_layer1_protectai.py --splits test --mode whole --out eval/results/l1_protectai_whole --blind
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("scripts"))
import evaluate_pipeline as ev  # noqa: E402
from eval_layer1_public import load  # noqa: E402

MODEL = "protectai/deberta-v3-base-prompt-injection-v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="dev")
    ap.add_argument("--mode", choices=["whole", "segment"], default="whole")
    ap.add_argument("--out", required=True)
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from app.rag.threat_gate import segment_text

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
    injection = [i for i, lab in model.config.id2label.items() if lab.upper() == "INJECTION"][0]

    rows = load(set(args.splits.split(",")))
    texts, owner = [], []
    for i, r in enumerate(rows):
        segs = segment_text(r["text"]) if args.mode == "segment" else [r["text"]]
        texts.extend(segs)
        owner.extend([i] * len(segs))
    probs = np.zeros(len(texts))
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(texts), args.batch):
            enc = tok(texts[start:start + args.batch], return_tensors="pt", padding=True, truncation=True, max_length=512)
            probs[start:start + args.batch] = torch.softmax(model(**enc).logits, dim=-1)[:, injection].numpy()
            if (start // args.batch) % 100 == 0:
                print(f"  {start + args.batch}/{len(texts)}  ({time.time() - t0:.0f}s)", flush=True)
    score = np.zeros(len(rows))
    np.maximum.at(score, np.array(owner), probs)

    results = [dict(r, action="BLOCK" if s >= 0.5 else "PASS", p_injection=round(float(s), 4)) for r, s in zip(rows, score)]
    os.makedirs(args.out, exist_ok=True)

    def table(key, title):
        groups = {}
        for r in results:
            groups.setdefault(key(r), []).append(r)
        out = [f"### {title}", "", "| Group | Attacks | Detected | Benign | Falsely detected |", "| --- | --- | --- | --- | --- |"]
        for k in sorted(groups):
            a = [x for x in groups[k] if x["label"] == "attack"]
            b = [x for x in groups[k] if x["label"] == "benign"]
            out.append(f"| {k} | {len(a)} | {ev.fmt_prop(sum(x['action'] == 'BLOCK' for x in a), len(a)) if a else '-'} | "
                       f"{len(b)} | {ev.fmt_prop(sum(x['action'] == 'BLOCK' for x in b), len(b)) if b else '-'} |")
        return out + [""]

    md = [f"# Baseline: {MODEL} ({args.mode})", "",
          f"splits {args.splits} · {len(rows)} rows · {len(texts)} model inputs · {time.time() - t0:.0f}s · commit `{ev.git_commit()}`", ""]
    md += table(lambda r: f"{r['dataset']} / {r['split']}", "Per dataset and split")
    md += table(lambda r: "ALL", "Pooled")
    md += table(lambda r: r["family"], "By family")
    md += table(lambda r: f"{r.get('lang_group', 'original')} / {r['lang']}", "By language group")
    open(os.path.join(args.out, "report.md"), "w", encoding="utf-8").write("\n".join(md))
    with open(os.path.join(args.out, "results.jsonl"), "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({k: v for k, v in r.items() if not (args.blind and k in ("text", "translated_from"))}, ensure_ascii=False) + "\n")
    print("\n".join(md[:12]))


if __name__ == "__main__":
    main()
