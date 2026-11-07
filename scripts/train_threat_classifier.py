#!/usr/bin/env python3
"""Trains the Layer 1 classifier (C) on the train split and calibrates its thresholds on the dev split.

The model is a logistic regression over the query embedding Layer 1 already computes, stored as plain numbers
(config/threat_classifier.json), never as a pickle. The test split is never read here.

Version 2 (default) differs from version 1 in three ways:
  * embedding model: paraphrase-multilingual-MiniLM-L12-v2 (50+ languages) instead of the English-only all-MiniLM-L6-v2;
  * training data: besides the public sets, the generated multilingual, social-engineering and long benign texts
    (scripts/build_layer1_multilingual.py, scripts/build_layer1_augmented.py). Every long benign text is also split
    into the segments the gate scores, each a benign example, so segment scoring does not flag ordinary long messages;
  * segment scoring: the gate scores the whole message AND every sentence / window (app/rag/threat_gate.segment_text)
    and keeps the highest probability. Thresholds are calibrated on dev with exactly that procedure, so the benign
    flag / block rates hold for the gate as it runs.
The block threshold is the lowest score at which the dev benign false-block rate stays under --block-fp (likewise the
flag threshold under --flag-fp). With --can-block the classifier may block alone above that threshold.

Usage: python scripts/train_threat_classifier.py                       # version 2
       python scripts/train_threat_classifier.py --version 1 --out config/threat_classifier_v1.json
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

from app.rag.threat_classifier import resolve_model  # noqa: E402
from app.rag.threat_gate import segment_text  # noqa: E402

V1_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
V2_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GENERATED = {"multilingual", "hidden_paragraph", "social_engineering"}   # files version 1 did not use


def load_rows(splits, include_generated):
    rows = []
    for path in sorted(glob.glob(os.path.join("eval", "external", "*.jsonl"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name in GENERATED and not include_generated:
            continue
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r["split"] in splits:
                r["dataset"] = name
                rows.append(r)
    return rows


class Embedder:
    """Normalised embeddings with an on-disk cache keyed by text hash (one cache file per model, git-ignored)."""

    def __init__(self, model_name):
        slug = model_name.split("/")[-1]
        self.cache_path = os.path.join("eval", "external", f"embeddings_{slug}.npz")
        self.model_name = model_name
        self.model = None
        self.cache = {}
        if os.path.exists(self.cache_path):
            z = np.load(self.cache_path, allow_pickle=False)
            self.cache = dict(zip(z["keys"].tolist(), z["vecs"]))
        self.dirty = False

    def __call__(self, texts):
        keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
        missing = sorted({k for k in keys if k not in self.cache})
        if missing:
            if self.model is None:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(resolve_model(self.model_name))
            by_key = {k: t for k, t in zip(keys, texts)}
            t0 = time.time()
            print(f"embedding {len(missing)} new texts with {self.model_name} ...", flush=True)
            for start in range(0, len(missing), 2000):
                chunk = missing[start:start + 2000]
                vecs = np.asarray(self.model.encode([by_key[k] for k in chunk], batch_size=64, show_progress_bar=False), dtype=np.float64)
                vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
                self.cache.update(zip(chunk, vecs))
                print(f"  {min(start + 2000, len(missing))}/{len(missing)}  ({time.time() - t0:.0f}s)", flush=True)
            self.dirty = True
        return np.stack([self.cache[k] for k in keys])

    def save(self):
        if self.dirty:
            ks = sorted(self.cache)
            np.savez(self.cache_path, keys=np.array(ks), vecs=np.stack([self.cache[k] for k in ks]))


def threshold_at(scores_benign, max_fp):
    """Lowest threshold whose benign false-positive rate is <= max_fp."""
    s = np.sort(scores_benign)
    k = int(np.floor(max_fp * len(s)))
    return float(s[-(k + 1)]) if len(s) else 1.0


def row_scores(rows, embed, prob_fn, segmented):
    """Score of each row as the gate computes it: whole text, or the max over its segments."""
    texts, owner = [], []
    for i, r in enumerate(rows):
        segs = segment_text(r["text"]) if segmented else [r["text"]]
        texts.extend(segs)
        owner.extend([i] * len(segs))
    probs = prob_fn(embed(texts))
    out = np.zeros(len(rows))
    np.maximum.at(out, np.array(owner), probs)
    return out


def calibrate_semantic(rows, embed, segmented, corpus_path="config/attack_corpus.yaml", benign_fp=0.02):
    """Similarity floor for V with the new model: at most `benign_fp` of dev benign rows get any V score."""
    import yaml
    phrases = (yaml.safe_load(open(corpus_path, encoding="utf-8")) or {}).get("phrases", [])
    attack_vecs = embed(phrases)
    benign = [r for r in rows if r["label"] == "benign"]
    sims = row_scores(benign, embed, lambda v: np.max(v @ attack_vecs.T, axis=1), segmented)
    floor = threshold_at(sims, benign_fp)
    return {"sim_floor": round(floor, 4), "sim_ceil": round(min(floor + 0.25, 0.99), 4), "dev_benign_fp_target": benign_fp}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", type=int, choices=[1, 2], default=2)
    ap.add_argument("--block-fp", type=float, default=0.005, help="max benign false-block rate on dev")
    ap.add_argument("--flag-fp", type=float, default=0.02, help="max benign false-flag rate on dev")
    ap.add_argument("--C", type=float, default=1.0, help="inverse L2 strength")
    ap.add_argument("--can-block", action=argparse.BooleanOptionalAction, default=None,
                    help="let the classifier block alone above the block threshold (default: on for version 2)")
    ap.add_argument("--out", default=os.path.join("config", "threat_classifier.json"))
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression

    v2 = args.version == 2
    model_name = V2_MODEL if v2 else V1_MODEL
    can_block = v2 if args.can_block is None else args.can_block
    embed = Embedder(model_name)

    train, dev = load_rows({"train"}, v2), load_rows({"dev"}, v2)
    # Training examples: every row as a whole text, except hidden-paragraph attacks (their attack sentence is already
    # a public training attack); plus, for version 2, every segment of every long benign text as a benign example.
    examples = [(r["text"], r["label"]) for r in train if not (r["dataset"] == "hidden_paragraph" and r["label"] == "attack")]
    if v2:
        for r in train:
            if r["label"] == "benign":
                segs = segment_text(r["text"])
                examples.extend((s, "benign") for s in segs[1:])
    examples = list(dict.fromkeys(examples))
    X = embed([t for t, _ in examples])
    y = np.array([1 if lab == "attack" else 0 for _, lab in examples])
    print(f"model {model_name} · train examples {len(examples)} (attacks {int(y.sum())}) · dev rows {len(dev)}", flush=True)

    clf = LogisticRegression(C=args.C, class_weight="balanced", max_iter=5000)
    clf.fit(X, y)
    prob = lambda vecs: clf.predict_proba(vecs)[:, 1]   # noqa: E731

    ydev = np.array([1 if r["label"] == "attack" else 0 for r in dev])
    p_dev = row_scores(dev, embed, prob, segmented=v2)
    benign, attack = p_dev[ydev == 0], p_dev[ydev == 1]
    t_block = float(np.nextafter(threshold_at(benign, args.block_fp), 1.0))
    t_flag = min(float(np.nextafter(threshold_at(benign, args.flag_fp), 1.0)), t_block)
    report = dict(
        dev_attacks=int(len(attack)), dev_benign=int(len(benign)),
        dev_attack_blocked=float((attack >= t_block).mean()), dev_attack_flagged_or_blocked=float((attack >= t_flag).mean()),
        dev_benign_blocked=float((benign >= t_block).mean()), dev_benign_flagged_or_blocked=float((benign >= t_flag).mean()))
    by_family = {}
    for fam in sorted({r.get("family", "?") for r in dev}):
        idx = [i for i, r in enumerate(dev) if r.get("family") == fam]
        lab = dev[idx[0]]["label"]
        hit = p_dev[idx] >= t_flag
        by_family[fam] = {"label": lab, "n": len(idx), "flagged_or_blocked": round(float(hit.mean()), 3)}
    print(json.dumps(dict(t_block=t_block, t_flag=t_flag, **report), indent=2))
    print(json.dumps(by_family, indent=1))

    semantic = calibrate_semantic(dev, embed, segmented=v2) if v2 else {}
    embed.save()

    manifest = json.load(open(os.path.join("eval", "external_manifest.json"), encoding="utf-8"))
    used = {k: v["sha256"][:16] for k, v in manifest["datasets"].items() if v2 or k not in GENERATED}
    cfg = dict(version=args.version, embedding_model=model_name, scoring="segment_max" if v2 else "whole",
               can_block=can_block, weights=[round(float(w), 6) for w in clf.coef_[0]],
               bias=round(float(clf.intercept_[0]), 6), threshold_flag=round(t_flag, 6), threshold_block=round(t_block, 6),
               training=dict(C=args.C, train_examples=len(examples), train_attacks=int(y.sum()), dev_calibration=report,
                             dev_by_family=by_family, target_dev_benign_block_fp=args.block_fp,
                             target_dev_benign_flag_fp=args.flag_fp, datasets=used,
                             note="trained on the train split; thresholds calibrated on dev with the gate's scoring; test split never read"))
    if semantic:
        cfg["semantic"] = semantic
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
