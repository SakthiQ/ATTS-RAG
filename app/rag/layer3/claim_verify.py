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


def split_sentences(text: str) -> List[str]:
    """Sentences of a passage. PDF text is hard-wrapped mid-sentence, so a newline only ends a sentence
    when the line ends in punctuation or is a table row / list item / heading-like line."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    merged: List[str] = []
    for ln in lines:
        prev = merged[-1] if merged else ""
        continues = (
            prev
            and not prev.endswith((".", "!", "?", ":", ";"))
            and not prev.startswith("|")
            and not ln.startswith(("|", "-", "*", "•"))
            and not re.match(r"\d+[.)]\s", ln)
            and " > " not in prev  # chunk breadcrumb, not prose
            and not re.match(r"[A-Z][\w ]{0,25}:\s", ln)  # a new "The Fix: ..." style label
            and len(prev.split()) > 3
        )
        if continues:
            merged[-1] = f"{prev} {ln}"
        else:
            merged.append(ln)
    out: List[str] = []
    for ln in merged:
        out.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", ln) if s.strip())
    return out


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

    def __init__(self, model_name: Optional[str] = None):
        self._nli_model = CrossEncoder(model_name or self.NLI_MODEL_NAME, max_length=512)
        # Models disagree on label order, so read it from the model config instead of assuming one
        id2label = {int(i): str(l).lower() for i, l in self._nli_model.model.config.id2label.items()}
        by_name = {label: i for i, label in id2label.items()}
        self._i_contra = by_name.get("contradiction", 0)
        self._i_entail = by_name.get("entailment", 1)

    # ------------------------------------------------------------------ windows
    def _windows(self, evidence_lookup: Dict[str, str]) -> List[Tuple[str, str]]:
        """All (evidence_id, premise) windows over the evidence."""
        out: List[Tuple[str, str]] = []
        for eid, text in evidence_lookup.items():
            sents = split_sentences(text)
            if len(text.split()) <= self.WHOLE_PASSAGE_MAX_WORDS:
                out.append((eid, text))
            out.extend((eid, sent) for sent in sents if len(sent.split()) >= 3)
            n = self.WINDOW_SENTENCES
            out.extend((eid, " ".join(sents[i:i + n])) for i in range(len(sents) - n + 1))
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
    def verify_claims(self, claims: List[ClaimObject], evidence_lookup: Dict[str, str]) -> List[ClaimVerificationResult]:
        results: List[Optional[ClaimVerificationResult]] = [None] * len(claims)
        windows = self._windows(evidence_lookup)
        all_numbers = set().union(*[_numbers(t) for t in evidence_lookup.values()]) if evidence_lookup else set()

        pairs: List[Tuple[str, str]] = []
        spans: List[Tuple[int, List[Tuple[str, str]]]] = []  # (claim index, its candidate windows)
        for idx, claim in enumerate(claims):
            missing = _numbers(claim.text) - all_numbers
            if missing:
                results[idx] = self._result(claim, "INSUFFICIENT", 0.0,
                                            f"Number(s) {sorted(missing)} do not appear in the evidence.")
                continue
            cands = self._candidates(claim, windows)
            extractive = self._extractive_match(claim, cands)
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
            results[idx] = self._decide(claim, cands, probs)
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

    def _decide(self, claim: ClaimObject, cands: List[Tuple[str, str]], probs: List[np.ndarray]) -> ClaimVerificationResult:
        entail = [float(p[self._i_entail]) for p in probs]
        contra = [float(p[self._i_contra]) for p in probs]
        best = int(np.argmax(entail))
        if entail[best] >= self.ENTAILMENT_THRESHOLD:
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
