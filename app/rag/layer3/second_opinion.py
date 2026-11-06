"""Quote-grounded second opinion for claims the NLI model could not confirm.

The small NLI verifier is conservative: it rejects true statements that restate a table row in other words, apply a
stated range ("4 years" inside "3 to 5 years"), or join two facts in one sentence. For those claims a second model of
a DIFFERENT family than the generator (so it does not share the generator's blind spots) is asked whether the evidence
supports the claim. Its answer is never trusted by itself. It must copy the supporting words from the evidence, and the
claim is released only if

  * the quote occurs verbatim in an evidence passage,
  * every number of the claim is in the quote (or lies inside a range the quote states),
  * the claim and the quote have the same number of negations ("may" can never become "may not"), and
  * the quote carries some of the claim's content words and the passage it comes from carries nearly all of them
    (a table lookup keeps its subject words in the headers, so the quote alone cannot be required to hold them).

A model that invents support therefore fails the quote check, and text planted in the evidence cannot make a false
claim pass, because none of these checks is decided by the model.
"""
import json
import re
from typing import Callable, Dict, List, Optional, Tuple

from loguru import logger

_WORD = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(r"\d[\d,]*\.?\d*")
_NEGATION = re.compile(r"\b(?:not|no|never|cannot|without|neither|nor)\b|n't", re.I)
_STOP = {"the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "was", "were", "for", "on", "that", "with", "as",
         "by", "it", "this", "be", "from", "at", "their", "its", "can", "has", "have"}

PROMPT = (
    "You are a strict fact checker. Decide whether the STATEMENT follows directly from the EXCERPTS below. "
    "Use only the excerpts. A statement follows if it restates what an excerpt says, picks one value out of a table row, "
    "or applies a range an excerpt gives (for example 4 years is inside '3 to 5 years'). It does not follow if it adds "
    "reasons, opinions, advice, comparisons or facts the excerpts do not state.\n"
    'Reply with JSON only: {{"verdict": "supported" or "not_supported", "excerpt": "E1", "quote": "words copied exactly from that excerpt"}}\n\n'
    "EXCERPTS:\n{excerpts}\n\nSTATEMENT: {claim}\n"
)


def _alnum(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def _terms(text: str) -> set:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


class QuoteGroundedJudge:
    """Wraps a text-in/text-out model call (any model family) behind the quote checks described above."""

    MIN_QUOTE_WORDS = 3
    MIN_QUOTE_COVERAGE = 0.3     # share of the claim's content words that the quote itself must contain
    MIN_PASSAGE_COVERAGE = 0.75  # share of the claim's content words that the quote's passage must contain
    MAX_EXCERPTS = 3

    def __init__(self, invoker: Callable[[str], str]):
        self.invoker = invoker
        self.calls = self.accepted = 0

    def check(self, claim: str, windows: List[Tuple[str, str]], passages: Dict[str, str],
              ranged: Optional[set] = None) -> Optional[Tuple[str, str]]:
        """(evidence_id, quote) if the claim is confirmed, else None. `windows` are (evidence_id, text) candidates."""
        from app.rag.layer3.claim_verify import _in_ranges, _numbers, _ranges   # deferred: claim_verify imports this module

        seen, excerpts = set(), []
        for eid, text in windows:
            if text not in seen and len(excerpts) < self.MAX_EXCERPTS:
                seen.add(text)
                excerpts.append((eid, text))
        if not excerpts:
            return None
        block = "\n".join(f"[{eid}] {text}" for eid, text in excerpts)
        self.calls += 1
        try:
            raw = self.invoker(PROMPT.format(excerpts=block, claim=claim))
            match = re.search(r"\{[\s\S]*\}", raw or "")
            verdict = json.loads(match.group(0)) if match else {}
        except Exception as exc:
            logger.warning(f"Second opinion unavailable: {exc}")
            return None
        if str(verdict.get("verdict", "")).lower() != "supported":
            return None
        quote = str(verdict.get("quote") or "").strip()
        if len(quote.split()) < self.MIN_QUOTE_WORDS:
            return None

        # 1. the quote must be real: it occurs in an evidence passage (the cited one if the model names it)
        wanted = str(verdict.get("excerpt") or "")
        order = [wanted] if wanted in passages else []
        order += [e for e in passages if e not in order]
        needle = _alnum(quote)
        eid = next((e for e in order if needle and needle in _alnum(passages[e])), None)
        if eid is None:
            return None
        # 2. numbers: each one in the quote, or inside a range the quote states
        ranged = ranged or set()
        claim_numbers = {n.replace(",", "").rstrip(".") for n in _NUMBER.findall(claim)}
        quote_numbers = _numbers(quote)
        quote_ranges = _ranges(quote)
        for n in claim_numbers:
            if n not in quote_numbers and not _in_ranges(n, quote_ranges):
                return None
        # 3. negation parity
        if len(_NEGATION.findall(claim)) != len(_NEGATION.findall(quote)):
            return None
        # 4. the quote and its passage must cover the claim's wording (reasons and opinions the passage never uses fail here)
        terms = _terms(claim) - ranged   # a number stated only as a range is checked above, not matched as a word
        if terms and (len(terms & _terms(quote)) / len(terms) < self.MIN_QUOTE_COVERAGE
                      or len(terms & _terms(passages[eid])) / len(terms) < self.MIN_PASSAGE_COVERAGE):
            return None
        self.accepted += 1
        return eid, quote
