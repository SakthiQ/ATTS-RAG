#!/usr/bin/env python3
"""Layer 1 delay per message, by message length: the old gate (version-1 classifier) against the current one.

Takes up to --per-bucket dev rows from each length bucket, screens each one with every gate (one message at a time, as
the API does) after a warm-up, and reports the median, 95th percentile and maximum time of ThreatGate.screen. Run it on
an otherwise idle machine: other jobs on the CPU inflate every number.

Usage: python scripts/measure_layer1_latency.py --out eval/results/l1_latency
"""
import argparse
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("scripts"))
import evaluate_pipeline as ev  # noqa: E402
from eval_layer1_public import load  # noqa: E402

BUCKETS = [(1, 20), (21, 50), (51, 150), (151, 300), (301, 10_000)]   # words
GATES = {"old gate (v1)": dict(classifier_path=os.path.join("config", "threat_classifier_v1.json"), use_claims=False),
         "new gate": dict()}


def pct(vals, q):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(round(q * (len(vals) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-bucket", type=int, default=100)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    ev.quiet_logs()
    rng = random.Random(args.seed)
    rows = load({"dev"})
    sample = []
    for lo, hi in BUCKETS:
        pool = [r for r in rows if lo <= len(r["text"].split()) <= hi]
        sample += [(f"{lo}-{hi if hi < 10_000 else '+'} words", r["text"]) for r in rng.sample(pool, min(args.per_bucket, len(pool)))]

    md = ["# Layer 1 delay per message", "",
          f"{len(sample)} dev messages · one at a time · times of ThreatGate.screen in ms · commit `{ev.git_commit()}`", "",
          "| Gate | Length | Messages | Median | 95th percentile | Max |", "| --- | --- | --- | --- | --- | --- |"]
    for name, flags in GATES.items():
        gate = ev.new_gate(use_rules=True, use_semantic=True, use_session=False, **flags)
        for _, text in sample[:20]:            # warm-up: model loading and first-call overhead
            gate.screen(text, user_id="warm", client_ip="10.0.0.1")
        times = {}
        for i, (bucket, text) in enumerate(sample):
            t0 = time.perf_counter()
            gate.screen(text, user_id=f"u{i}", client_ip="10.0.0.1")
            times.setdefault(bucket, []).append((time.perf_counter() - t0) * 1000.0)
        everything = [t for ts in times.values() for t in ts]
        for bucket, ts in list(times.items()) + [("all", everything)]:
            md.append(f"| {name} | {bucket} | {len(ts)} | {statistics.median(ts):.0f} | {pct(ts, 0.95):.0f} | {max(ts):.0f} |")
        print(f"done: {name}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
