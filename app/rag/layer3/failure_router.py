from typing import List, Literal, Tuple, Dict, Optional
from app.rag.layer3.contracts import ClaimVerificationResult, SafetyCheckResult, RelevanceResult


class FailureTypeRouter:
    """Classifies verification failures according to whether the problem is generation behavior or an evidence deficiency."""

    def route_failure(
        self,
        safety_res: SafetyCheckResult,
        relevance_res: RelevanceResult,
        claim_results: List[ClaimVerificationResult],
        retry_count: int,
        has_usable_evidence: bool = True
    ) -> Tuple[Literal["PASS", "RETRY", "REJECT"], str]:
        """Returns action: PASS, RETRY (1-shot targeted retry), or REJECT (immediate exit)."""
        # 1. High-confidence secret leak -> Immediate REJECT (No retry makes leaked secret safe)
        if safety_res.leaked_secrets_found:
            return "REJECT", "High-confidence secret leakage detected in output."

        # 2. Schema or Invalid Evidence ID failure
        if not safety_res.passed:
            if not has_usable_evidence:
                return "REJECT", "Evidence gap: No usable supporting evidence in Layer 2 package."
            if retry_count < 1 and safety_res.invalid_evidence_ids:
                return "RETRY", f"Invalid evidence IDs cited: {safety_res.invalid_evidence_ids}. Requesting retry."
            elif retry_count < 1:
                return "RETRY", f"Schema validation failed: {safety_res.reason}. Requesting retry."
            else:
                return "REJECT", f"Schema validation failed after retry: {safety_res.reason}"

        # 3. Off-topic answer relevance failure
        if not relevance_res.passed:
            if not has_usable_evidence:
                return "REJECT", "Evidence gap: No usable supporting evidence in Layer 2 package."
            if retry_count < 1:
                return "RETRY", f"Draft answer is irrelevant to question: {relevance_res.reason}"
            else:
                return "REJECT", f"Answer failed relevance check: {relevance_res.reason}"

        # 4. Analyze Claim Grounding Failures
        unknowns = [c for c in claim_results if c.status == "UNKNOWN"]
        contradictions = [c for c in claim_results if c.status == "CONTRADICTION"]
        insufficient = [c for c in claim_results if c.status == "INSUFFICIENT"]
        entailed = [c for c in claim_results if c.status == "ENTAILMENT"]

        # Fail closed on UNKNOWN status (verifier failure / timeout)
        if unknowns:
            return "REJECT", f"Verifier produced UNKNOWN status for claims: {[c.claim_id for c in unknowns]} (fail-closed security boundary)."

        # Hard rejection/retry on direct contradiction
        if contradictions:
            if not has_usable_evidence:
                return "REJECT", "Evidence gap: Verified Layer 2 package lacks supporting evidence."
            if retry_count < 1:
                return "RETRY", f"Contradiction detected in claims: {[c.claim_id for c in contradictions]}"
            return "REJECT", f"Direct contradiction detected in claims: {[c.claim_id for c in contradictions]}"

        # If all claims passed, PASS directly
        if not insufficient and entailed:
            return "PASS", "All claims passed verification."

        # If partial claims passed with no contradictions, PASS to allow Reconstructor to isolate verified claims
        if entailed:
            return "PASS", f"{len(entailed)} claim(s) passed verification; reconstructing answer from verified subset."

        # Check evidence availability: if package lacks usable evidence, reject without retry
        if not has_usable_evidence:
            return "REJECT", "Evidence gap: Verified Layer 2 package lacks supporting evidence."

        # If no claims passed and retry_count < 1, allow 1-shot targeted retry
        if retry_count < 1:
            return "RETRY", f"Insufficient evidence support for claims: {[c.claim_id for c in insufficient]}"

        # If retry attempt exhausted and 0 claims passed
        return "REJECT", "Claims failed grounding verification and retry attempts exhausted."

