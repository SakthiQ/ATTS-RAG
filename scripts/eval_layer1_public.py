#!/usr/bin/env python3
"""Layer 1 on public prompt-injection / jailbreak / benign datasets (fetched by scripts/fetch_external_data.py).

Each text is screened by a fresh gate state (no session memory, rate limit off), so the numbers measure the detector
itself. Reports, per dataset and split: attacks blocked, attacks flagged-or-blocked, benign blocked, benign
flagged-or-blocked, with Wilson 95% intervals, plus a per-language breakdown.

Discipline: tune on --splits train,dev only. The `test` split is run once for the final report; with --blind the
script prints no individual texts, so looking at a run cannot leak test examples into the next tuning round.

Usage: python scripts/eval_layer1_public.py --splits dev --out eval/results/l1_public_baseline
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("scripts"))
import evaluate_pipeline as ev  # noqa: E402


def load(splits):
    rows = []
    for path in sorted(glob.glob(os.path.join("eval", "external", "*.jsonl"))):
        name = os.path.splitext(os.path.basename(path))[0]
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r["split"] in splits:
                r["dataset"] = name
                rows.append(r)
    return rows


def pct(k, n):
    return ev.fmt_prop(k, n) if n else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="dev", help="comma list of train,dev,test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--blind", action="store_true", help="do not write misclassified texts (use for the test split)")
    ap.add_argument("--config", default="full", choices=["baseline", "v1", "full", "full-can-block", "classifier-only"],
                    help="baseline = rules + semantic only; v1 = the version-1 classifier (English MiniLM, whole-message "
                         "scoring, flag-only) from config/threat_classifier_v1.json; full = the current config/threat_classifier.json")
    ap.add_argument("--max-words", type=int, default=0, help="skip texts longer than this many words (0 = keep all)")
    args = ap.parse_args()
    ev.quiet_logs()
    os.makedirs(args.out, exist_ok=True)

    splits = set(args.splits.split(","))
    rows = load(splits)
    if args.max_words:
        rows = [r for r in rows if len(r["text"].split()) <= args.max_words]
    flags = {"baseline": dict(use_classifier=False), "v1": dict(classifier_path=os.path.join("config", "threat_classifier_v1.json")),
             "full": dict(), "full-can-block": dict(classifier_can_block=True),
             "classifier-only": dict(use_rules=False, use_semantic=False)}[args.config]
    gate = ev.new_gate(**{**dict(use_rules=True, use_semantic=True, use_session=False), **flags})

    started, results = time.time(), []
    for i, r in enumerate(rows):
        res = gate.screen(r["text"], user_id=f"u{i}", tenant_id="eval", client_ip="10.0.0.1")
        results.append(dict(r, action=res.action, base_risk=res.base_risk, scores=res.detector_scores))
        if (i + 1) % 1000 == 0:
            print(f"  screened {i + 1}/{len(rows)}  ({time.time() - started:.0f}s)", flush=True)

    def table(group_key, title):
        groups = defaultdict(list)
        for r in results:
            groups[group_key(r)].append(r)
        out = [f"### {title}", "", "| Group | Attacks | Blocked | Flagged or blocked | Benign | Blocked (FP) | Flagged or blocked |",
               "| --- | --- | --- | --- | --- | --- | --- |"]
        for key in sorted(groups):
            g = groups[key]
            a = [x for x in g if x["label"] == "attack"]
            b = [x for x in g if x["label"] == "benign"]
            out.append(f"| {key} | {len(a)} | {pct(sum(x['action'] == 'BLOCK' for x in a), len(a))} | "
                       f"{pct(sum(x['action'] != 'PASS' for x in a), len(a))} | {len(b)} | "
                       f"{pct(sum(x['action'] == 'BLOCK' for x in b), len(b))} | {pct(sum(x['action'] != 'PASS' for x in b), len(b))} |")
        return out + [""]

    md = ["# Layer 1 on public datasets", "",
          f"config: {args.config} · splits: {sorted(splits)} · {len(results)} texts · {time.time() - started:.0f}s · commit `{ev.git_commit()}`", ""]
    md += table(lambda r: f"{r['dataset']} / {r['split']}", "Per dataset and split")
    md += table(lambda r: "ALL", "Pooled")
    md += table(lambda r: r["lang"], "By language (guessed)")
    md += table(lambda r: r["family"], "By family")
    md += table(lambda r: f"{r.get('lang_group', 'original')} / {r['lang']}", "By language group (multilingual set: seen in training or not)")
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(os.path.join(args.out, "results.jsonl"), "w", encoding="utf-8") as f:
        for r in results:
            slim = {k: v for k, v in r.items() if not (args.blind and k in ("text", "translated_from"))}
            f.write(json.dumps(slim, ensure_ascii=False) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
