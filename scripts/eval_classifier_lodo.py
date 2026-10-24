#!/usr/bin/env python3
"""Leave-one-dataset-out check of the Layer 1 classifier: does it work on a dataset it never saw?

For each public dataset D: train on the TRAIN rows of all other datasets, calibrate thresholds on the DEV rows of those
same other datasets (same rule as the real classifier), then score ALL rows of D. Datasets that are attack-only
(gandalf) or benign-only (dolly, jbb) still contribute, as the other side of every fold. Uses cached embeddings, so it
needs no model run. Reads only train/dev rows of the training datasets; D's own test rows are scored but not used to choose
anything.

Usage: python scripts/eval_classifier_lodo.py --out eval/results/classifier_lodo
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("scripts"))
import train_threat_classifier as T  # noqa: E402


def ci(k, n):
    from evaluate_pipeline import wilson
    p, lo, hi = wilson(k, n)
    return f"{100 * p:.1f}% [{100 * lo:.0f}-{100 * hi:.0f}] ({k}/{n})" if n else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--block-fp", type=float, default=0.005)
    ap.add_argument("--flag-fp", type=float, default=0.02)
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    os.makedirs(args.out, exist_ok=True)

    rows = T.load_rows({"train", "dev", "test"})
    X = T.embeddings_for(rows)
    y = np.array([1 if r["label"] == "attack" else 0 for r in rows])
    ds = np.array([r["dataset"] for r in rows])
    sp = np.array([r["split"] for r in rows])
    md = ["# Classifier: leave-one-dataset-out", "",
          "Trained without the named dataset; thresholds calibrated on the other datasets' dev rows; every row of the "
          "named dataset is scored.", "",
          "| Held-out dataset | Attacks blocked | Attacks flagged or blocked | Benign | Benign blocked (FP) | Benign flagged or blocked |",
          "| --- | --- | --- | --- | --- | --- |"]
    for held in sorted(set(ds)):
        tr = (ds != held) & (sp == "train")
        dv = (ds != held) & (sp == "dev")
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(X[tr], y[tr])
        pdv = clf.predict_proba(X[dv])[:, 1]
        t_block = float(np.nextafter(T.threshold_at(pdv[y[dv] == 0], args.block_fp), 1.0))
        t_flag = min(float(np.nextafter(T.threshold_at(pdv[y[dv] == 0], args.flag_fp), 1.0)), t_block)
        m = ds == held
        p = clf.predict_proba(X[m])[:, 1]
        a, b = p[y[m] == 1], p[y[m] == 0]
        md.append(f"| {held} | {ci(int((a >= t_block).sum()), len(a))} | {ci(int((a >= t_flag).sum()), len(a))} | {len(b)} | "
                  f"{ci(int((b >= t_block).sum()), len(b))} | {ci(int((b >= t_flag).sum()), len(b))} |")
    text = "\n".join(md)
    open(os.path.join(args.out, "report.md"), "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
