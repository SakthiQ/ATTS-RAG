#!/usr/bin/env python3
"""Builds a multilingual attack / benign set for Layer 1 by machine-translating rows of the public data.

The public prompt-injection data is almost all English, so a multilingual detector cannot be trained or measured on it.
This script translates short attack and benign texts with NLLB-200 (facebook/nllb-200-distilled-600M, CC-BY-NC-4.0:
research use only) and keeps the original split of every row, so a test sentence is never a translation of a training
sentence.

Languages
  seen    (train, dev and test): French, German, Spanish, Hindi
  unseen  (test only):           Italian, Portuguese, Chinese (Simplified)
The unseen languages measure whether detection carries over to languages the classifier never saw in training.

Limits: machine translations are cleaner than real multilingual attacks and can soften or garble an instruction, so
these rows measure robustness to language, not real-world multilingual attack rates.

Output (git-ignored): eval/external/multilingual.jsonl, with the translated text, the original split and label,
`lang` (NLLB code) and `translated_from` (the English source). Resumable: rows already written are skipped.
Usage: python scripts/build_layer1_multilingual.py [--per-split train=400,dev=100,test=150]
"""
import argparse
import glob
import hashlib
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("scripts"))

OUT = os.path.join("eval", "external", "multilingual.jsonl")
MODEL = "facebook/nllb-200-distilled-600M"
SEEN = ["fra_Latn", "deu_Latn", "spa_Latn", "hin_Deva"]
UNSEEN = ["ita_Latn", "por_Latn", "zho_Hans"]
SOURCES = {"attack": ("deepset_prompt_injections", "gandalf_ignore_instructions", "safeguard_prompt_injection"),
           "benign": ("dolly_benign", "domain_benign", "deepset_prompt_injections", "safeguard_prompt_injection")}
MAX_WORDS = 40
SEED = 29


def load_pool(split, label):
    rows = []
    for name in SOURCES[label]:
        path = os.path.join("eval", "external", f"{name}.jsonl")
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r["split"] == split and r["label"] == label and r.get("lang", "en") == "en" and len(r["text"].split()) <= MAX_WORDS:
                rows.append(dict(r, dataset=name))
    rows.sort(key=lambda r: hashlib.sha1(r["text"].encode("utf-8")).hexdigest())   # stable before sampling
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-split", default="train=400,dev=100,test=150", help="rows per label per split")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    per_split = {k: int(v) for k, v in (p.split("=") for p in args.per_split.split(","))}

    done = set()
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["source_id"], r["lang"]))

    jobs = []   # (source row, target language)
    rng = random.Random(SEED)
    for split, n in per_split.items():
        langs = SEEN + (UNSEEN if split == "test" else [])
        for label in ("attack", "benign"):
            pool = load_pool(split, label)
            for r in rng.sample(pool, min(n, len(pool))):
                sid = hashlib.sha1(r["text"].encode("utf-8")).hexdigest()[:16]
                for lang in langs:
                    if (sid, lang) not in done:
                        jobs.append((dict(r, source_id=sid), lang))
    print(f"{len(jobs)} translations to do ({len(done)} already written)", flush=True)
    if not jobs:
        return

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    model.eval()

    t0 = time.time()
    with open(OUT, "a", encoding="utf-8") as out:
        for lang in SEEN + UNSEEN:
            batch_jobs = [j for j in jobs if j[1] == lang]
            for start in range(0, len(batch_jobs), args.batch):
                chunk = batch_jobs[start:start + args.batch]
                enc = tok([r["text"] for r, _ in chunk], return_tensors="pt", padding=True, truncation=True, max_length=128)
                gen = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(lang), max_new_tokens=128, num_beams=1)
                for (r, _), text in zip(chunk, tok.batch_decode(gen, skip_special_tokens=True)):
                    out.write(json.dumps({
                        "text": text, "label": r["label"], "split": r["split"],
                        "family": f"translated_{r['label']}", "lang": lang,
                        "lang_group": "seen" if lang in SEEN else "unseen",
                        "source": f"nllb:{r['dataset']}", "source_id": r["source_id"], "translated_from": r["text"],
                    }, ensure_ascii=False) + "\n")
                out.flush()
                done_n = start + len(chunk)
                if done_n % (args.batch * 10) == 0 or done_n == len(batch_jobs):
                    print(f"  {lang}: {done_n}/{len(batch_jobs)}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
