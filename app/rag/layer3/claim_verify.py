import re
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from app.rag.layer3.contracts import ClaimObject, ClaimVerificationResult


class DirectClaimVerifier:
    """Direct Claim-to-Cited-Evidence NLI Engine (O(N) Complexity).
    Verifies claims against the exact evidence chunks cited in their evidence_ids
    using a local cross-encoder NLI model (DeBERTa-v3-small).
    """

    NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
    ENTAILMENT_THRESHOLD = 0.70
    CONTRADICTION_THRESHOLD = 0.50

    def __init__(self):
        """Loads the NLI model once at initialization time."""
        self._nli_model = CrossEncoder(self.NLI_MODEL_NAME, max_length=512)

    def verify_claims(
        self,
        claims: List[ClaimObject],
        evidence_lookup: Dict[str, str]
    ) -> List[ClaimVerificationResult]:
        """Runs NLI verification for each claim against its cited evidence text."""
        results = []

        for claim in claims:
            if not claim.evidence_ids:
                results.append(
                    ClaimVerificationResult(
                        claim_id=claim.claim_id,
                        text=claim.text,
                        cited_evidence_ids=[],
                        status="INSUFFICIENT",
                        reason="Claim contains no evidence citations."
                    )
                )
                continue

            # Gather cited evidence content
            cited_texts = []
            missing_ids = []
            for eid in claim.evidence_ids:
                if eid in evidence_lookup:
                    cited_texts.append(evidence_lookup[eid])
                else:
                    missing_ids.append(eid)

            if missing_ids:
                results.append(
                    ClaimVerificationResult(
                        claim_id=claim.claim_id,
                        text=claim.text,
                        cited_evidence_ids=claim.evidence_ids,
                        status="INSUFFICIENT",
                        reason=f"Cited evidence IDs not found in package: {missing_ids}"
                    )
                )
                continue

            combined_evidence = " ".join(cited_texts)

            # Perform NLI classification with fail-closed exception handling
            try:
                status, confidence, reason = self._evaluate_nli(claim.text, combined_evidence)
            except Exception as exc:
                status, confidence, reason = "UNKNOWN", 0.0, f"Verifier execution error: {str(exc)}"

            results.append(
                ClaimVerificationResult(
                    claim_id=claim.claim_id,
                    text=claim.text,
                    cited_evidence_ids=claim.evidence_ids,
                    status=status,
                    confidence=confidence,
                    reason=reason
                )
            )

        return results

    def _evaluate_nli(self, claim_text: str, evidence_text: str) -> tuple[str, float, str]:
        """Evaluates semantic relationship between claim and evidence using DeBERTa-v3-small NLI.

        DeBERTa-v3-small NLI label ordering: [contradiction, entailment, neutral]
        Returns raw logits which are converted to probabilities via softmax.
        """
        scores = self._nli_model.predict([(evidence_text, claim_text)])
        # scores shape: (1, 3) or (3,) depending on model version
        raw = scores[0] if hasattr(scores[0], '__len__') and len(scores[0]) == 3 else scores
        probs = self._softmax(raw)

        contradiction_score = float(probs[0])
        entailment_score = float(probs[1])
        neutral_score = float(probs[2])

        if entailment_score >= self.ENTAILMENT_THRESHOLD:
            return "ENTAILMENT", entailment_score, "Evidence supports claim."
        elif contradiction_score >= self.CONTRADICTION_THRESHOLD:
            return "CONTRADICTION", contradiction_score, "Evidence contradicts claim."
        else:
            return (
                "INSUFFICIENT",
                max(entailment_score, neutral_score),
                f"Insufficient evidence support (entail={entailment_score:.2f}, contra={contradiction_score:.2f}, neutral={neutral_score:.2f})"
            )

    @staticmethod
    def _softmax(logits) -> np.ndarray:
        """Converts raw logits to normalized probabilities."""
        arr = np.array(logits, dtype=np.float64)
        e = np.exp(arr - np.max(arr))
        return e / e.sum()
