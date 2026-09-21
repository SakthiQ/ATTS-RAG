from typing import List
from app.rag.layer3.contracts import ClaimObject, ClaimVerificationResult


class Layer3PolicyEnforcer:
    """Enforces policy-tiered blocking rules for released answers."""

    def evaluate_policy(
        self,
        policy_class: str,
        verified_claims: List[ClaimObject],
        failed_claims: List[ClaimObject],
        verification_results: List[ClaimVerificationResult]
    ) -> tuple[bool, str]:
        """Returns (is_policy_satisfied, reason)."""
        if not verified_claims:
            return False, "Zero claims passed verification."

        if policy_class == "SECURITY_CRITICAL":
            if failed_claims:
                return False, f"SECURITY_CRITICAL policy rejects response with unverified claims: {[c.claim_id for c in failed_claims]}"

        return True, "Policy requirements satisfied."
