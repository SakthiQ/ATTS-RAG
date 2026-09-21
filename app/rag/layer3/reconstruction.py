from typing import List, Tuple
from app.rag.layer3.contracts import ClaimObject, ClaimVerificationResult


class VerifiedAnswerReconstructor:
    """Reconstructs released answer exclusively from claims that passed verification."""

    def reconstruct(
        self,
        claims: List[ClaimObject],
        verification_results: List[ClaimVerificationResult]
    ) -> Tuple[str, List[ClaimObject], List[ClaimObject]]:
        """Filters to verified claims and composes the final released response string."""
        passed_ids = {
            res.claim_id for res in verification_results if res.status == "ENTAILMENT"
        }

        verified_claims = []
        failed_claims = []

        # Maintain claim order
        sorted_claims = sorted(claims, key=lambda c: c.order)

        for claim in sorted_claims:
            if claim.claim_id in passed_ids:
                verified_claims.append(claim)
            else:
                failed_claims.append(claim)

        if not verified_claims:
            return "", [], failed_claims

        # Format answer text with citations
        formatted_blocks = []
        for claim in verified_claims:
            cites = ", ".join(claim.evidence_ids) if claim.evidence_ids else ""
            cite_str = f" [{cites}]" if cites else ""
            formatted_blocks.append(f"{claim.text}{cite_str}")

        reconstructed_answer = "\n".join(formatted_blocks)
        return reconstructed_answer, verified_claims, failed_claims
