#!/usr/bin/env python3
"""Downloads small public prompt-injection / jailbreak / benign-prompt datasets and normalises them.

Output (git-ignored): eval/external/<name>.jsonl, one JSON object per line:
  {"text", "label": "attack"|"benign", "family", "lang", "source", "split": "train"|"dev"|"test"}
The manifest eval/external_manifest.json (committed) records the source, counts and a hash of every file.

Splits are fixed here, before any tuning: an official test split is used as `test`; the official train split is cut
once (seed 13) into `train` (80%) and `dev` (20%). Detectors are trained and tuned on train / dev only; `test` is for
the single final report. Datasets without an official split are cut 70/15/15 with the same seed.

Sources (all public, openly licensed): deepset/prompt-injections (apache-2.0), xTRam1/safe-guard-prompt-injection,
jackhhao/jailbreak-classification (apache-2.0), Lakera/gandalf_ignore_instructions (mit),
JailbreakBench/JBB-Behaviors (mit; benign half only), databricks/databricks-dolly-15k (cc-by-sa-3.0; a sample).

Usage: python scripts/fetch_external_data.py
"""
import hashlib
import json
import os
import random
import re
from collections import Counter

OUT_DIR = os.path.join("eval", "external")
MANIFEST = os.path.join("eval", "external_manifest.json")
SEED = 13
DOLLY_SAMPLE = 1500

_STOP = {
    "en": {"the", "and", "is", "are", "you", "to", "of", "in", "that", "for", "with", "what", "how", "your", "all"},
    "de": {"der", "die", "das", "und", "ist", "nicht", "ich", "sie", "mit", "den", "ein", "eine", "zu", "von", "wie", "was", "alle"},
    "fr": {"le", "la", "les", "et", "est", "pas", "vous", "des", "une", "que", "pour", "dans", "ignorez", "toutes"},
    "es": {"el", "los", "las", "y", "es", "una", "que", "para", "por", "con", "todas", "instrucciones"},
}


def guess_lang(text: str) -> str:
    """Crude stop-word language guess (en / de / fr / es / other); good enough to report results per language."""
    words = set(re.findall(r"[a-zA-ZäöüßéèàçñÄÖÜ]+", text.lower()))
    scores = {lang: len(words & stop) for lang, stop in _STOP.items()}
    best = max(scores, key=scores.get)
    if scores[best] >= 2:
        return best
    return "en" if re.fullmatch(r"[\x00-\x7f]*", text) else "other"


def cut(rows, seed, fractions):
    """Splits rows into named parts by the given fractions (the last part takes the remainder)."""
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    parts, start = {}, 0
    names = list(fractions)
    for i, name in enumerate(names):
        end = len(rows) if i == len(names) - 1 else start + round(fractions[name] * len(rows))
        parts[name] = rows[start:end]
        start = end
    return parts


def build(source, items_by_split, family_of=None):
    """items_by_split: {"train": [(text, label)], "test": [...]} -> normalised rows with a dev cut from train."""
    out = []
    train_parts = cut(items_by_split.get("train", []), SEED, {"train": 0.8, "dev": 0.2}) if "train" in items_by_split else {}
    parts = {**train_parts}
    for name in ("validation",):
        if name in items_by_split:   # folded into dev
            parts["dev"] = parts.get("dev", []) + items_by_split[name]
    if "test" in items_by_split:
        parts["test"] = items_by_split["test"]
    for split, items in parts.items():
        for text, label in items:
            text = str(text).strip()
            if text:
                out.append(dict(text=text, label=label, family=family_of(text, label) if family_of else label,
                                lang=guess_lang(text), source=source, split=split))
    return out


def write(name, rows, manifest):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    manifest[name] = dict(file=path.replace("\\", "/"), rows=len(rows), sha256=digest,
                          by_split_label={f"{s}/{l}": n for (s, l), n in sorted(Counter((r["split"], r["label"]) for r in rows).items())},
                          by_lang=dict(Counter(r["lang"] for r in rows)))
    print(f"{name:34s} {len(rows):6d} rows  {manifest[name]['by_split_label']}")


def main():
    from datasets import load_dataset
    manifest = {}

    ds = load_dataset("deepset/prompt-injections")
    write("deepset_prompt_injections", build(
        "deepset/prompt-injections",
        {s: [(r["text"], "attack" if int(r["label"]) == 1 else "benign") for r in ds[s]] for s in ("train", "test")}), manifest)

    ds = load_dataset("xTRam1/safe-guard-prompt-injection")
    write("safeguard_prompt_injection", build(
        "xTRam1/safe-guard-prompt-injection",
        {s: [(r["text"], "attack" if int(r["label"]) == 1 else "benign") for r in ds[s]] for s in ("train", "test")}), manifest)

    ds = load_dataset("jackhhao/jailbreak-classification")
    write("jailbreak_classification", build(
        "jackhhao/jailbreak-classification",
        {s: [(r["prompt"], "attack" if r["type"] == "jailbreak" else "benign") for r in ds[s]] for s in ("train", "test")},
        family_of=lambda t, l: "jailbreak_prompt" if l == "attack" else "benign"), manifest)

    ds = load_dataset("Lakera/gandalf_ignore_instructions")
    write("gandalf_ignore_instructions", build(
        "Lakera/gandalf_ignore_instructions",
        {s: [(r["text"], "attack") for r in ds[s]] for s in ("train", "validation", "test")},
        family_of=lambda t, l: "direct_override"), manifest)

    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    benign = [(r["Goal"], "benign") for r in ds["benign"]]
    parts = cut(benign, SEED, {"train": 0.5, "test": 0.5})
    write("jbb_benign", build("JailbreakBench/JBB-Behaviors (benign)", {"train": parts["train"], "test": parts["test"]},
                              family_of=lambda t, l: "hard_negative"), manifest)

    ds = load_dataset("databricks/databricks-dolly-15k")["train"]
    sample = random.Random(SEED).sample(range(len(ds)), DOLLY_SAMPLE)
    items = [(ds[i]["instruction"], "benign") for i in sample]
    parts = cut(items, SEED, {"train": 0.7, "dev": 0.15, "test": 0.15})
    write("dolly_benign", [dict(text=t.strip(), label=l, family="benign", lang=guess_lang(t), source="databricks/databricks-dolly-15k",
                                split=s) for s, its in parts.items() for t, l in its if t.strip()], manifest)

    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "note": "data files are git-ignored; rebuild with scripts/fetch_external_data.py", "datasets": manifest}, f, indent=2)
    print(f"\nManifest written to {MANIFEST}")


if __name__ == "__main__":
    main()
