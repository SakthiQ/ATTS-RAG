from typing import List, Set, Dict, Any
from app.rag.layer3.contracts import GenerationContract, SafetyCheckResult


class SchemaValidator:
    """Validates structural integrity of structured JSON generation contracts and evidence allow-lists."""

    def validate(
        self,
        contract: GenerationContract,
        allowed_evidence_ids: Set[str]
    ) -> SafetyCheckResult:
        """Validates that:
        1. Claims list is non-empty.
        2. Claim IDs are unique.
        3. All cited evidence IDs exist in the Layer 2 allowed evidence set.
        """
        if not contract.claims:
            return SafetyCheckResult(
                passed=False,
                reason="Generation contract contains no claims."
            )

        invalid_ids = []
        claim_ids = set()

        for claim in contract.claims:
            if not claim.text or not claim.text.strip():
                return SafetyCheckResult(
                    passed=False,
                    reason=f"Claim {claim.claim_id} contains empty text."
                )

            if claim.claim_id in claim_ids:
                return SafetyCheckResult(
                    passed=False,
                    reason=f"Duplicate claim ID detected: {claim.claim_id}"
                )
            claim_ids.add(claim.claim_id)

            for eid in claim.evidence_ids:
                if allowed_evidence_ids and eid not in allowed_evidence_ids:
                    invalid_ids.append(eid)

        if invalid_ids:
            return SafetyCheckResult(
                passed=False,
                reason=f"Cited evidence IDs not in Layer 2 allow-list: {list(set(invalid_ids))}",
                invalid_evidence_ids=list(set(invalid_ids))
            )

        return SafetyCheckResult(passed=True)
