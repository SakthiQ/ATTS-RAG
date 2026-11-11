"""Text normalisation and the injection-pattern feed, shared by Layer 1, the upload scan and Layer 2.

Patterns are written for plain Latin text, so every scanner first undoes the cheap disguises: compatibility forms
(NFKC), invisible format characters such as zero-width spaces, and Cyrillic/Greek look-alike letters.
"""
import re
import unicodedata
from typing import Any, Dict, List

import yaml
from loguru import logger

# Cyrillic/Greek letters that render like Latin ones. NFKC leaves them alone, so
# "Ignоre" with a Cyrillic "о" would otherwise slip past every Latin regex.
HOMOGLYPHS = str.maketrans({
    # Cyrillic
    "а": "a", "в": "b", "е": "e", "к": "k", "м": "m", "н": "h", "о": "o", "р": "p",
    "с": "c", "т": "t", "у": "y", "х": "x", "ѕ": "s", "і": "i", "ј": "j", "ԁ": "d",
    "ԛ": "q", "ԝ": "w", "ү": "y", "һ": "h",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P",
    "С": "C", "Т": "T", "У": "Y", "Х": "X", "Ѕ": "S", "І": "I", "Ј": "J",
    # Greek
    "α": "a", "ε": "e", "ι": "i", "κ": "k", "ν": "v", "ο": "o", "ρ": "p", "τ": "t",
    "υ": "u", "χ": "x",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M",
    "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
})


def fold_homoglyphs(text: str) -> str:
    """Maps look-alike Cyrillic/Greek letters to Latin where they are disguising Latin text.

    Folds everything when every non-ASCII letter is a look-alike (the foreign script is
    pure disguise); otherwise folds only words that mix ASCII letters with look-alikes.
    Real Cyrillic/Greek text almost always has letters with no Latin twin (ч, л, π, ή...),
    so genuine non-Latin text is left untouched.
    """
    foreign = [ch for ch in text if ch.isalpha() and not ch.isascii()]
    if not foreign:
        return text
    if all(ord(ch) in HOMOGLYPHS for ch in foreign):
        return text.translate(HOMOGLYPHS)
    return re.sub(
        r"\w+",
        lambda m: m.group(0).translate(HOMOGLYPHS)
        if any(ch.isascii() and ch.isalpha() for ch in m.group(0)) else m.group(0),
        text,
    )


def strip_control_chars(text: str) -> str:
    """Removes control and format characters (zero-width spaces, joiners, direction marks), keeping line breaks."""
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\r", "\t"))


def normalize_for_scan(text: str) -> str:
    """The text a pattern scanner should read. Never stored or shown: only used for matching."""
    return fold_homoglyphs(strip_control_chars(unicodedata.normalize("NFKC", text or "")))


def load_patterns(path: str = "config/injection_patterns.yaml") -> List[Dict[str, Any]]:
    """Loads the threat feed. A missing file is an error: a scan must not run without patterns."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    flags = re.IGNORECASE | re.DOTALL | re.MULTILINE
    patterns = [
        {"id": p["id"], "severity": float(p["severity"]), "regex": re.compile(p["regex"], flags)}
        for p in config["patterns"]
    ]
    logger.info(f"Loaded {len(patterns)} injection patterns from {path}")
    return patterns
