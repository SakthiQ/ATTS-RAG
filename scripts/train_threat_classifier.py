#!/usr/bin/env python3
"""Trains the small Layer 1 classifier on the public train split and calibrates its thresholds on the dev split.

The model is a logistic regression over the same all-MiniLM-L6-v2 query embedding Layer 1 already computes, stored
as plain numbers (config/threat_classifier.json), never as a pickle. Thresholds are NOT guessed: the block threshold
is the lowest score at which the DEV benign false-block rate stays under --block-fp, the flag threshold likewise under
--flag-fp. The test split is never read here.

Usage: python scripts/train_threat_classifier.py --block-fp 0.005 --flag-fp 0.02
Needs scikit-learn (a development dependency only; the runtime classifier uses numpy).
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath("."))

EMB_CACHE = os.path.join("eval", "external", "embeddings_minilm.npz")
OUT = os.path.join("config", "threat_classifier.json")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_rows(splits):
    rows = []
    for path in sorted(glob.glob(os.path.join("eval", "external", "*.jsonl"))):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r["split"] in splits:
                r["dataset"] = os.path.splitext(os.path.basename(path))[0]
                rows.append(r)
    return rows


def embed(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    vecs = np.asarray(model.encode(texts, batch_size=64, show_progress_bar=False), dtype=np.float64)
    return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)


def embeddings_for(rows):
    """Embeddings of the rows' texts, cached on disk by text hash (the cache is git-ignored)."""
    cache = {}
    if os.path.exists(EMB_CACHE):
        z = np.load(EMB_CACHE, allow_pickle=False)
        cache = dict(zip(z["keys"].tolist(), z["vecs"]))
    keys = [hashlib.sha1(r["text"].encode("utf-8")).hexdigest() for r in rows]
    missing = sorted({k for k in keys if k not in cache})
    if missing:
        by_key = {k: r["text"] for k, r in zip(keys, rows)}
        t0 = time.time()
        print(f"embedding {len(missing)} new texts ...", flush=True)
        for start in range(0, len(missing), 2000):
            chunk = missing[start:start + 2000]
            for k, v in zip(chunk, embed([by_key[k] for k in chunk])):
                cache[k] = v
            print(f"  {min(start + 2000, len(missing))}/{len(missing)}  ({time.time() - t0:.0f}s)", flush=True)
        ks = sorted(cache)
        np.savez(EMB_CACHE, keys=np.array(ks), vecs=np.stack([cache[k] for k in ks]))
    return np.stack([cache[k] for k in keys])


def threshold_at(scores_benign, max_fp):
    """Lowest threshold whose benign false-positive rate is <= max_fp."""
    s = np.sort(scores_benign)
    k = int(np.floor(max_fp * len(s)))
    return float(s[-(k + 1)]) if len(s) else 1.0   # allow k benign items at or above the threshold... then step above


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-fp", type=float, default=0.005, help="max benign false-block rate on dev")
    ap.add_argument("--flag-fp", type=float, default=0.02, help="max benign false-flag rate on dev")
    ap.add_argument("--C", type=float, default=1.0, help="inverse L2 strength")
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression

    train, dev = load_rows({"train"}), load_rows({"dev"})
    Xtr, Xdev = embeddings_for(train), embeddings_for(dev)
    ytr = np.array([1 if r["label"] == "attack" else 0 for r in train])
    ydev = np.array([1 if r["label"] == "attack" else 0 for r in dev])
    print(f"train {len(train)} (attacks {ytr.sum()}) · dev {len(dev)} (attacks {ydev.sum()})")

    clf = LogisticRegression(C=args.C, class_weight="balanced", max_iter=3000)
    clf.fit(Xtr, ytr)
    p_dev = clf.predict_proba(Xdev)[:, 1]
    benign, attack = p_dev[ydev == 0], p_dev[ydev == 1]
    t_block = float(np.nextafter(threshold_at(benign, args.block_fp), 1.0))
    t_flag = min(float(np.nextafter(threshold_at(benign, args.flag_fp), 1.0)), t_block)
    report = dict(
        dev_attacks=int(len(attack)), dev_benign=int(len(benign)),
        dev_attack_blocked=float((attack >= t_block).mean()), dev_attack_flagged_or_blocked=float((attack >= t_flag).mean()),
        dev_benign_blocked=float((benign >= t_block).mean()), dev_benign_flagged_or_blocked=float((benign >= t_flag).mean()))
    print(json.dumps(dict(t_block=t_block, t_flag=t_flag, **report), indent=2))

    manifest = json.load(open(os.path.join("eval", "external_manifest.json"), encoding="utf-8"))
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(dict(
            version=1, embedding_model=EMBEDDING_MODEL, weights=[round(float(w), 6) for w in clf.coef_[0]],
            bias=round(float(clf.intercept_[0]), 6), threshold_flag=round(t_flag, 6), threshold_block=round(t_block, 6),
            training=dict(C=args.C, train_rows=len(train), train_attacks=int(ytr.sum()), dev_calibration=report,
                          target_dev_benign_block_fp=args.block_fp, target_dev_benign_flag_fp=args.flag_fp,
                          datasets={k: v["sha256"][:16] for k, v in manifest["datasets"].items()},
                          note="trained on the public train split; thresholds calibrated on dev; test split never read")),
            f, indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
