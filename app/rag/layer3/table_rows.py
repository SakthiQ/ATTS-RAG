"""Deterministic check for claims that state values from the rows of a Markdown table.

The NLI model reads one linearized row at a time ("Plan: Business; Monthly price per seat: $9; ...") and often scores a
correct paraphrase of it as neutral ("The Business plan costs $9 per seat per month"), or even as a contradiction when
the row is a range ("100% to 149% | 8%" for "a representative at 120% of quota earns 8%"). Correct table facts were the
largest fixable group of correct sentences the verifier rejected, so they are checked here on the table itself.

A claim passes only if all of these hold:
  * it has no negation;
  * every number in it is accounted for: it is a value in the row the claim names (or inside a range cell of that row),
    or it selects a row whose label is a range ("120%" in "100% to 149%"), or, before the first row is named, it
    appears in a prose sentence of the same passage that shares the claim's words ("Q3 revenue was $95.2 million");
  * each value comes from a column whose header shares a word with the claim, with the same unit ("15 minutes" is not
    "15 hours");
  * values are paired with rows by position: the nearest row named before the value, or in order with "respectively";
  * the claim adds at most one content word that is not in the rows used, the headers or the passage.
Anything else goes back to the NLI model, so this check can only accept claims, never reject them.
"""
import re
from typing import Dict, List, Optional, Set, Tuple

from app.rag.layer3.table_check import parse_tables

_WORD = re.compile(r"[a-z0-9]+")
_NUM = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
_NEGATION = re.compile(r"\b(?:not|no|never|cannot|without|neither|nor|none|except)\b|n't", re.I)
_STOP = {"the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "was", "were", "for", "on", "that", "with", "as",
         "by", "it", "this", "be", "from", "at", "their", "its", "has", "have", "had", "which", "than", "all", "any",
         "each", "per", "respectively", "while", "whereas", "both", "also", "they", "these", "those", "there", "within",
         "get", "gets", "will", "can", "may", "must", "up"}
_UNITS = {"minute", "hour", "day", "week", "month", "year", "seat", "percent", "million", "billion", "thousand"}
_SCALE = {"million", "billion", "thousand"}   # "$42.1 million" may quote a cell of "42.1" (USD millions)
MAX_NEW_WORDS = 1          # content words of the claim that the table and its passage do not contain
CONTEXT_COVERAGE = 0.6     # a number stated before any row is named must sit in a prose sentence sharing this share of words


def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("ly"):
        return word[:-2]
    if len(word) > 4 and word.endswith(("ches", "shes", "xes", "sses")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _words(text: str) -> Set[str]:
    """Content stems without numbers."""
    return {_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP and not w[0].isdigit()}


def _num(text: str) -> float:
    return float(text.replace(",", ""))


def _ranges(text: str) -> List[Tuple[float, float]]:
    from app.rag.layer3.claim_verify import _ranges as ranges   # one definition of "a range" for the whole verifier
    return ranges(text)


def _unit_after(text: str, end: int) -> Optional[str]:
    m = re.match(r"\s*(%|[a-z]+)", text[end:].lower())
    if not m:
        return None
    unit = "percent" if m.group(1) == "%" else _stem(m.group(1))
    return unit if unit in _UNITS else None


def _money(text: str, pos: int) -> bool:
    return text[max(0, pos - 1):pos] == "$"


def _clause(text: str, pos: int) -> str:
    """The comma- or semicolon-delimited part of `text` around position `pos`."""
    start = max(text.rfind(",", 0, pos), text.rfind(";", 0, pos)) + 1
    ends = [i for i in (text.find(",", pos), text.find(";", pos)) if i != -1]
    return text[start:min(ends) if ends else len(text)]


def _is_range_label(label: str) -> bool:
    return bool(_ranges(label))


def _label_spans(low_claim: str, label: str) -> List[Tuple[int, int]]:
    words = _WORD.findall(label.lower())
    if not words or not any(w[0].isalpha() for w in words):
        return []
    pattern = r"\b" + r"[\W_]+".join(re.escape(w) for w in words) + r"s?\b"
    return [(m.start(), m.end()) for m in re.finditer(pattern, low_claim)]


def _value_in_row(number: str, unit: Optional[str], row: List[str], headers: List[str],
                  claim_words: Set[str], money: bool = False) -> Optional[str]:
    """The header of the cell of `row` that states `number` (literally or inside a range), if that cell fits the claim."""
    value = _num(number)
    numeric_cells = [j for j, c in enumerate(row) if j > 0 and _NUM.search(c)]
    for j, cell in enumerate(row):
        if j == 0:
            continue
        literal = [m for m in _NUM.finditer(cell) if _num(m.group(0)) == value]
        in_range = any(lo <= value <= hi for lo, hi in _ranges(cell))
        if not literal and not in_range:
            continue
        header = headers[j] if j < len(headers) else ""
        if not (_words(header) & claim_words) and len(numeric_cells) > 1:
            continue   # the claim does not talk about this column, and the row has others it could mean
        spans = [(m.start(), m.end()) for m in literal] or [(m.start(), m.end()) for m in _NUM.finditer(cell)][-1:]
        cell_unit = _unit_after(cell, spans[0][1]) if spans else None
        cell_money = bool(spans) and cell[max(0, spans[0][0] - 1):spans[0][0]] == "$"
        if money and cell_unit and cell_unit not in _SCALE:
            continue   # "$12" is not "12 seats"
        if cell_money and unit and unit not in _SCALE:
            continue   # "9 seats" is not "$9"
        if unit and cell_unit and cell_unit != unit:
            continue
        return header or f"column {j + 1}"
    return None


def _check_table(claim: str, headers: List[str], rows: List[List[str]], prose: List[str],
                 passage_words: Set[str]) -> Optional[str]:
    claim_words = _words(claim)
    low = claim.lower()
    # rows named in words ("Business", "P1", "Q2 2026")
    mentions: List[Tuple[int, int, int]] = []   # (start, end, row index)
    for i, row in enumerate(rows):
        if row and not _is_range_label(row[0]):
            mentions += [(s, e, i) for s, e in _label_spans(low, row[0])]
    taken = [(s, e) for s, e, _ in mentions]
    numbers = [(m.start(), m.group(0)) for m in _NUM.finditer(claim) if not any(s <= m.start() < e for s, e in taken)]
    if not numbers:
        return None
    # rows selected by a number inside a range label ("120%" selects "100% to 149%")
    values = {_num(m) for row in rows for c in row[1:] for m in _NUM.findall(c)}
    remaining = []
    for pos, n in numbers:
        hit = [i for i, row in enumerate(rows) if row and _is_range_label(row[0])
               and any(lo <= _num(n) <= hi for lo, hi in _ranges(row[0]))]
        if len(hit) == 1 and _num(n) not in values:
            mentions.append((pos, pos + len(n), hit[0]))
        else:
            remaining.append((pos, n))
    mentions.sort()
    used_rows = {i for _, _, i in mentions}
    checked: List[Tuple[str, int, str]] = []   # (number, row, column)

    if not mentions:
        # a list of one column's values ("the rates are 2%, 5%, 8% and 12%")
        if len(remaining) < 2:
            return None
        for j in range(1, len(headers)):
            if not (_words(headers[j]) & claim_words):
                continue
            rows_hit = []
            for _, n in remaining:
                r = [i for i, row in enumerate(rows) if j < len(row) and any(_num(m) == _num(n) for m in _NUM.findall(row[j]))]
                if len(r) != 1:
                    break
                rows_hit.append(r[0])
            else:
                if len(set(rows_hit)) == len(rows_hit):
                    used_rows = set(rows_hit)
                    checked = [(n, i, headers[j]) for (_, n), i in zip(remaining, rows_hit)]
                    break
        if not checked:
            return None
    else:
        first = mentions[0][0]
        order: List[int] = []
        for _, _, i in mentions:
            if i not in order:
                order.append(i)
        after_last = [(p, n) for p, n in remaining if p > mentions[-1][0]]
        respectively = "respectively" in low and len(order) > 1 and after_last and len(after_last) % len(order) == 0
        assign: Dict[int, int] = {}
        if respectively:
            k = len(after_last) // len(order)
            for t, (p, _) in enumerate(after_last):
                assign[p] = order[t // k]
        for p, _ in remaining:
            if p in assign:
                continue
            before = [i for s, _, i in mentions if s < p]
            # stated before any row is named: a value of the single named row, or context from the passage's prose
            assign[p] = before[-1] if before else -1
        for p, n in remaining:
            i = assign[p]
            if i == -1:
                # a single named row may also hold it ("The plan for 40 seats is the Business plan")
                if len(order) == 1:
                    column = _value_in_row(n, _unit_after(claim, p + len(n)), rows[order[0]], headers, claim_words, _money(claim, p))
                    if column is not None:
                        checked.append((n, order[0], column))
                        continue
                seg_words = _words(_clause(claim[:first], p))
                if not any(_num(n) in {_num(x) for x in _NUM.findall(s)}
                           and seg_words and len(seg_words & _words(s)) / len(seg_words) >= CONTEXT_COVERAGE for s in prose):
                    return None
                continue
            column = _value_in_row(n, _unit_after(claim, p + len(n)), rows[i], headers, claim_words, _money(claim, p))
            if column is None:
                return None
            checked.append((n, i, column))
        if not checked:
            return None   # the claim names rows but states none of their values

    known = set(passage_words)
    for i in used_rows:
        known |= _words(" ".join(rows[i]))
    known |= _words(" ".join(headers))
    if len(claim_words - known) > MAX_NEW_WORDS:
        return None
    pairs = "; ".join(f"{n} = {rows[i][0]} / {col}" for n, i, col in checked[:4])
    return f"Table row check: {pairs}."


def check_table_row_claim(claim: str, passages: Dict[str, str]) -> Optional[Tuple[str, str]]:
    """(evidence_id, reason) if every value the claim states comes from the table rows it names, else None."""
    if _NEGATION.search(claim) or not _NUM.search(claim):
        return None
    for eid, text in passages.items():
        tables = parse_tables(text)
        if not tables:
            continue
        prose_lines = [ln.strip() for ln in text.split("\n") if ln.strip() and not ln.strip().startswith("|")]
        prose = [s for ln in prose_lines for s in re.split(r"(?<=[.!?])\s+", ln) if s]
        passage_words = _words(" ".join(prose_lines))
        for headers, rows in tables:
            if len(headers) < 2 or not rows:
                continue
            reason = _check_table(claim, headers, rows, prose, passage_words)
            if reason:
                return eid, reason
    return None
