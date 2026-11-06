"""Deterministic check for comparative claims about Markdown tables ("North America had the highest revenue").

A sentence-level NLI model sees one table row at a time, so it cannot confirm a superlative: "highest" is a
statement about every row. This module parses the tables in an evidence passage and checks the claim directly:
the row named in the claim must hold the maximum (or minimum) of the column the claim talks about, and the
numbers the claim quotes must come from that row. A claim that adds words the passage does not contain is not
accepted here; it goes back to the normal verifier.
"""
import re
from typing import Dict, List, Optional, Tuple

_WORD = re.compile(r"[a-z0-9]+")
_NUM = re.compile(r"\d[\d,]*\.?\d*")
_STOP = {"the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "was", "were", "for", "on", "that", "with", "as",
         "by", "it", "this", "be", "from", "at", "their", "its", "has", "have", "had", "which", "than", "all", "any"}
_MAX = {"highest", "largest", "greatest", "biggest", "most", "maximum", "longest", "slowest", "top"}
_MIN = {"lowest", "smallest", "least", "fewest", "minimum", "shortest", "fastest", "quickest", "cheapest"}
_RANGE_CELL = re.compile(r"\d\s*(?:-|–|—|to)\s*\d", re.IGNORECASE)
_UNIT_MINUTES ={"minute": 1, "minutes": 1, "hour": 60, "hours": 60, "day": 1440, "days": 1440, "week": 10080, "weeks": 10080}
MIN_COVERAGE = 0.85   # share of the claim's words that must occur in the passage (or be the superlative / a number)


def _norm_num(text: str) -> str:
    return text.replace(",", "").rstrip(".")


def _terms(text: str) -> set:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def _cell_value(cell: str) -> Optional[float]:
    """First number in a table cell, converted to minutes when a time unit follows it."""
    if _RANGE_CELL.search(cell):
        return None   # "1-2 weeks" has no single value; comparing its first number with "1 week" would give wrong extremes
    m = _NUM.search(cell)
    if not m:
        return None
    try:
        value = float(_norm_num(m.group(0)))
    except ValueError:
        return None
    unit = re.match(r"\s*([a-z]+)", cell[m.end():].lower())
    return value * _UNIT_MINUTES.get(unit.group(1), 1) if unit else value


def parse_tables(text: str) -> List[Tuple[List[str], List[List[str]]]]:
    """(headers, rows) for each Markdown pipe table in the text."""
    lines = [ln.strip() for ln in text.split("\n")]
    tables, i = [], 0
    while i < len(lines):
        if lines[i].startswith("|") and i + 1 < len(lines) and re.fullmatch(r"\|?[\s:|-]+\|?", lines[i + 1]) and "-" in lines[i + 1]:
            headers = [c.strip() for c in lines[i].strip("|").split("|")]
            rows, i = [], i + 2
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            tables.append((headers, rows))
        else:
            i += 1
    return tables


def check_table_claim(claim: str, passages: Dict[str, str]) -> Optional[Tuple[str, str]]:
    """(evidence_id, reason) if the claim is a superlative that a table in the evidence confirms, else None."""
    words = _terms(claim)
    direction = "max" if words & _MAX else "min" if words & _MIN else None
    if direction is None:
        return None
    claim_numbers = {_norm_num(n) for n in _NUM.findall(claim)}
    for eid, text in passages.items():
        passage_terms = _terms(text)
        # numbers outside the tables (titles, captions): they may be quoted next to a row's own numbers
        outside_numbers = {_norm_num(n) for ln in text.split("\n") if not ln.strip().startswith("|") for n in _NUM.findall(ln)}
        for headers, rows in parse_tables(text):
            # the row the claim is about: a non-numeric cell of the row occurs in the claim
            low = claim.lower()
            subject = [r for r in rows if any(c and not _NUM.search(c) and c.lower() in low for c in r)]
            if len(subject) != 1:
                continue
            row = subject[0]
            header_terms = [_terms(h) for h in headers]
            for j, header in enumerate(headers):
                if j >= len(row):
                    continue
                values = [_cell_value(r[j]) for r in rows if j < len(r)]
                if len(values) != len(rows) or any(v is None for v in values) or len(rows) < 2:
                    continue
                mine = _cell_value(row[j])
                quoted = {_norm_num(n) for n in _NUM.findall(row[j])} & claim_numbers
                if not (header_terms[j] & words or quoted):
                    continue   # the claim is not about this column
                best = max(values) if direction == "max" else min(values)
                if mine != best or values.count(best) != 1:
                    continue
                # every number the claim quotes must come from this row or the passage's headings
                row_numbers = {_norm_num(n) for c in row for n in _NUM.findall(c)}
                if not claim_numbers <= (row_numbers | outside_numbers):
                    continue
                allowed = passage_terms | _MAX | _MIN | {_norm_num(n) for n in claim_numbers}
                if len(words & allowed) / len(words) < MIN_COVERAGE:
                    continue
                return eid, f"Table check: '{row[0]}' holds the {'highest' if direction == 'max' else 'lowest'} value of '{header}'."
    return None
