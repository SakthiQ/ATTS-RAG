import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from sentence_transformers import CrossEncoder

from app.rag.layer3.contracts import ClaimObject, ClaimVerificationResult

_WORD = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(r"\d[\d,]*\.?\d*")
_NEGATION = re.compile(r"\b(?:not|no|never|cannot|without|neither|nor)\b|n't", re.I)
_STOP = {"the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "was", "were", "for", "on", "that",
         "with", "as", "by", "it", "this", "be", "from", "at", "their", "its", "can", "has", "have"}


def _linearize_tables(lines: List[str]) -> List[Tuple[str, bool]]:
    """Turns Markdown pipe tables into one sentence per row ("Severity: P1; Target time: 15 minutes.").

    An NLI model cannot read "| P1 | Active breach | 15 minutes |", so claims about table rows were rejected
    even when they were exactly right. Pairing each cell with its column header makes the row a statement.
    Returns (line, is_table_row); rows must never be merged with their neighbours (see _windows).
    """
    out: List[Tuple[str, bool]] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("|") and i + 1 < len(lines) and re.fullmatch(r"\|?[\s:|-]+\|?", lines[i + 1]) and "-" in lines[i + 1]:
            headers = [c.strip() for c in lines[i].strip("|").split("|")]
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                out.append(("; ".join(f"{h}: {c}" for h, c in zip(headers, cells) if c) + ".", True))
                i += 1
        else:
            out.append((lines[i], False))
            i += 1
    return out


def split_sentences_marked(text: str) -> List[Tuple[str, bool]]:
    """Sentences of a passage, each flagged True if it is a linearized table row.

    PDF text is hard-wrapped mid-sentence, so a newline only ends a sentence when the line ends in
    punctuation or is a table row / list item / heading-like line.
    """
    lines = _linearize_tables([ln.strip() for ln in text.split("\n") if ln.strip()])
    merged: List[Tuple[str, bool]] = []
    for ln, is_row in lines:
        prev = merged[-1][0] if merged else ""
        continues = (
            not is_row
            and prev
            and not merged[-1][1]
            and not prev.endswith((".", "!", "?", ":", ";"))
            and not prev.startswith("|")
            and not ln.startswith(("|", "-", "*", "\u2022"))
            and not re.match(r"\d+[.)]\s", ln)
            and " > " not in prev  # chunk breadcrumb, not prose
            and not re.match(r"[A-Z][\w ]{0,25}:\s", ln)  # a new "The Fix: ..." style label
            and len(prev.split()) > 3
        )
        if continues:
            merged[-1] = (f"{prev} {ln}", False)
        else:
            merged.append((ln, is_row))
    out: List[Tuple[str, bool]] = []
    for ln, is_row in merged:
        if is_row:
            out.append((ln, True))
        else:
            out.extend((s.strip(), False) for s in re.split(r"(?<=[.!?])\s+", ln) if s.strip())
    return out


def split_sentences(text: str) -> List[str]:
    return [s for s, _ in split_sentences_marked(text)]


def _numbers(text: str) -> set:
    return {n.replace(",", "").rstrip(".") for n in _NUMBER.findall(text)}


def _terms(text: str) -> set:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


class DirectClaimVerifier:
    """Checks every claim against the evidence with a local NLI cross-encoder.

    A claim is compared with short windows of evidence (two consecutive sentences, or a whole short
    passage) rather than with whole passages, because the NLI model truncates at 512 tokens and
    long premises dilute the entailment signal. The best-supporting window decides the verdict.
    Two deterministic guards run around the model: every number in a claim must appear in the
    evidence, and a claim that cites the wrong passage is re-attributed to the passage that
    actually supports it.
    """

    NLI_MODEL_NAME = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
    ENTAILMENT_THRESHOLD = 0.60
    CONTRADICTION_THRESHOLD = 0.60
    MAX_WINDOWS = 12
    WINDOW_SENTENCES = 2
    WHOLE_PASSAGE_MAX_WORDS = 90
    EXTRACTIVE_COVERAGE = 0.9
    # Switches for ablation studies; production keeps both guards on
    use_number_guard = True
    use_extractive = True

    def __init__(self, model_name: Optional[str] = None):
        self._nli_model = CrossEncoder(model_name or self.NLI_MODEL_NAME, max_length=512)
        # Models disagree on label order, so read it from the model config instead of assuming one
        id2label = {int(i): str(l).lower() for i, l in self._nli_model.model.config.id2label.items()}
        by_name = {label: i for i, label in id2label.items()}
        self._i_contra = by_name.get("contradiction", 0)
        self._i_entail = by_name.get("entailment", 1)

    # ------------------------------------------------------------------ windows
    def _windows(self, evidence_lookup: Dict[str, str]) -> List[Tuple[str, str]]:
        """All (evidence_id, premise) windows over the evidence.

        Table rows are independent facts: pairing two rows (or taking a whole table as one premise) lets the
        NLI model entail "P1 ... 3 days" from "P1 ... 15 minutes. P4 ... 3 days", so rows only form single windows.
        """
        out: List[Tuple[str, str]] = []
        for eid, text in evidence_lookup.items():
            marked = split_sentences_marked(text)
            sents = [s for s, _ in marked]
            has_rows = any(is_row for _, is_row in marked)
            if len(text.split()) <= self.WHOLE_PASSAGE_MAX_WORDS and not has_rows:
                out.append((eid, text))
            out.extend((eid, sent) for sent in sents if len(sent.split()) >= 3)
            n = self.WINDOW_SENTENCES
            for i in range(len(marked) - n + 1):
                group = marked[i:i + n]
                if not any(is_row for _, is_row in group):
                    out.append((eid, " ".join(s for s, _ in group)))
        return out

    def _candidates(self, claim: ClaimObject, windows: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """The windows most likely to support the claim: lexical overlap, with cited passages favoured."""
        claim_terms = _terms(claim.text)
        cited = set(claim.evidence_ids)

        def score(w: Tuple[str, str]) -> float:
            overlap = len(claim_terms & _terms(w[1])) / (len(claim_terms) or 1)
            # Shorter windows win ties: the NLI model is distracted by sentences unrelated to the claim
            return overlap + (0.15 if w[0] in cited else 0.0) - 0.0005 * len(w[1].split())

        return sorted(windows, key=score, reverse=True)[:self.MAX_WINDOWS]

    # ------------------------------------------------------------------ verification
    def verify_claims(self, claims: List[ClaimObject], evidence_lookup: Dict[str, str],
                      entailment_threshold: Optional[float] = None) -> List[ClaimVerificationResult]:
        """`entailment_threshold` overrides the default for this call (an adaptive policy raises it for suspicious users)."""
        threshold = entailment_threshold if entailment_threshold is not None else self.ENTAILMENT_THRESHOLD
        results: List[Optional[ClaimVerificationResult]] = [None] * len(claims)
        windows = self._windows(evidence_lookup)
        all_numbers = set().union(*[_numbers(t) for t in evidence_lookup.values()]) if evidence_lookup else set()

        pairs: List[Tuple[str, str]] = []
        spans: List[Tuple[int, List[Tuple[str, str]]]] = []  # (claim index, its candidate windows)
        for idx, claim in enumerate(claims):
            missing = (_numbers(claim.text) - all_numbers) if self.use_number_guard else set()
            if missing:
                results[idx] = self._result(claim, "INSUFFICIENT", 0.0,
                                            f"Number(s) {sorted(missing)} do not appear in the evidence.")
                continue
            cands = self._candidates(claim, windows)
            extractive = self._extractive_match(claim, cands) if self.use_extractive else None
            if extractive is not None:
                results[idx] = extractive
                continue
            if not cands:
                results[idx] = self._result(claim, "INSUFFICIENT", 0.0, "No evidence available.")
                continue
            spans.append((idx, cands))
            pairs.extend((premise, claim.text) for _, premise in cands)

        if not pairs:
            return [r for r in results if r is not None]

        try:
            logits = self._nli_model.predict(pairs, batch_size=32)
        except Exception as exc:
            logger.error(f"NLI verifier failed: {exc}")
            for idx, _ in spans:
                results[idx] = self._result(claims[idx], "UNKNOWN", 0.0, f"Verifier execution error: {exc}")
            return results  # type: ignore[return-value]

        cursor = 0
        for idx, cands in spans:
            claim = claims[idx]
            probs = [self._softmax(row) for row in logits[cursor:cursor + len(cands)]]
            cursor += len(cands)
            results[idx] = self._decide(claim, cands, probs, threshold)
        return results  # type: ignore[return-value]

    def _extractive_match(self, claim: ClaimObject, cands: List[Tuple[str, str]]) -> Optional[ClaimVerificationResult]:
        """Near-verbatim claims (and list items copied from the evidence) pass a deterministic check.

        NLI cannot judge fragments such as "Neglecting Quick Wins (50% of failures)", and it often
        mislabels faithful quotations with unusual layout as contradictions. A claim therefore passes
        here only if at least EXTRACTIVE_COVERAGE of its content words and every number appear in a
        single evidence window and the window has the same number of negations (so "may" cannot
        become "may not").
        """
        terms, numbers = _terms(claim.text), _numbers(claim.text)
        if len(terms) < 2:
            return None
        negations = len(_NEGATION.findall(claim.text))
        for eid, premise in cands:
            if (numbers <= _numbers(premise)
                    and len(terms & _terms(premise)) / len(terms) >= self.EXTRACTIVE_COVERAGE
                    and len(_NEGATION.findall(premise)) == negations):
                return self._result(claim, "ENTAILMENT", 0.9, "Extractive match: wording and numbers found in the evidence.", eid)
        return None

    def _decide(self, claim: ClaimObject, cands: List[Tuple[str, str]], probs: List[np.ndarray],
                threshold: Optional[float] = None) -> ClaimVerificationResult:
        threshold = threshold if threshold is not None else self.ENTAILMENT_THRESHOLD
        entail = [float(p[self._i_entail]) for p in probs]
        contra = [float(p[self._i_contra]) for p in probs]
        best = int(np.argmax(entail))
        if entail[best] >= threshold:
            support_id = cands[best][0]
            note = "Evidence supports claim."
            if claim.evidence_ids and support_id not in claim.evidence_ids:
                note = f"Evidence supports claim; citation corrected to {support_id}."
            return self._result(claim, "ENTAILMENT", entail[best], note, support_id)
        claim_terms = _terms(claim.text)
        # A "contradiction" from a passage that barely mentions the claim's topic is noise, not evidence
        topical = [len(claim_terms & _terms(c[1])) / (len(claim_terms) or 1) >= 0.4 for c in cands]
        worst = int(np.argmax([c if t else 0.0 for c, t in zip(contra, topical)]))
        if topical[worst] and contra[worst] >= self.CONTRADICTION_THRESHOLD:
            return self._result(claim, "CONTRADICTION", contra[worst], "Evidence contradicts claim.", cands[worst][0])
        return self._result(claim, "INSUFFICIENT", entail[best],
                            f"Insufficient evidence support (best entailment {entail[best]:.2f}).")

    @staticmethod
    def _result(claim: ClaimObject, status: str, confidence: float, reason: str,
                support_id: Optional[str] = None) -> ClaimVerificationResult:
        return ClaimVerificationResult(
            claim_id=claim.claim_id, text=claim.text, cited_evidence_ids=claim.evidence_ids,
            status=status, confidence=round(confidence, 4), reason=reason, supporting_evidence_id=support_id,
        )

    @staticmethod
    def _softmax(logits) -> np.ndarray:
        arr = np.array(logits, dtype=np.float64)
        e = np.exp(arr - np.max(arr))
        return e / e.sum()
