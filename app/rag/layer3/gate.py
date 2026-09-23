import asyncio
import time
from typing import List, Dict, Any, Optional, Callable, Set

from app.rag.layer3.contracts import (
    Layer3GateDecision,
    GenerationContract,
    SafetyCheckResult,
    RelevanceResult,
    ClaimVerificationResult
)
from app.rag.layer3.generation import ContractConstrainedGenerator
from app.rag.layer3.schema import SchemaValidator
from app.rag.layer3.safety import FastFailSafetyScanner
from app.rag.layer3.relevance import RelevanceEvaluator
from app.rag.layer3.claim_verify import DirectClaimVerifier
from app.rag.layer3.failure_router import FailureTypeRouter
from app.rag.layer3.reconstruction import VerifiedAnswerReconstructor
from app.rag.layer3.policy import Layer3PolicyEnforcer
from app.rag.layer3.telemetry import Layer3TelemetryStore


class Layer3Gate:
    """Master Layer 3 Evidence-to-Answer & Output Verification Gate.
    Enforces contract-constrained generation, fast-fail safety, parallel relevance and direct claim verification,
    failure routing, and verified answer reconstruction.
    """

    def __init__(
        self,
        llm_invoker: Optional[Callable[[str], str]] = None,
        model_version: str = "generator-v1"
    ):
        self.generator = ContractConstrainedGenerator(llm_invoker=llm_invoker)
        self.schema_validator = SchemaValidator()
        self.safety_scanner = FastFailSafetyScanner()
        self.relevance_evaluator = RelevanceEvaluator()
        self.claim_verifier = DirectClaimVerifier()
        self.failure_router = FailureTypeRouter()
        self.reconstructor = VerifiedAnswerReconstructor()
        self.policy_enforcer = Layer3PolicyEnforcer()
        self.telemetry_store = Layer3TelemetryStore()
        self.model_version = model_version

    async def process(
        self,
        question: str,
        layer2_evidence_package: List[Dict[str, Any]],
        query_id: str = "Q-101",
        policy_class: str = "ORDINARY_INFORMATIONAL",
        tenant_id: Optional[str] = None
    ) -> Layer3GateDecision:
        """Executes full Layer 3 pipeline: Gen -> Fast Safety -> Parallel (Relevance + Direct NLI) -> Router -> Reconstruction."""
        start_time = time.time()

        # Defense-in-depth Cross-Tenant Isolation Check
        if tenant_id:
            for item in layer2_evidence_package:
                item_tenant = item.get("metadata", {}).get("tenant_id")
                if item_tenant and item_tenant != tenant_id:
                    return self._build_reject_decision(
                        query_id=query_id,
                        reason=f"Cross-tenant authorization violation: evidence chunk tenant '{item_tenant}' does not match query tenant '{tenant_id}'.",
                        retry_count=0,
                        start_time=start_time
                    )

        # Build evidence lookup map & allowed evidence ID set from Layer 2
        allowed_evidence_ids: Set[str] = set()
        evidence_lookup: Dict[str, str] = {}

        for idx, item in enumerate(layer2_evidence_package):
            meta = item.get("metadata", {})
            eid = meta.get("chunk_uid") or f"E{idx+1}"
            allowed_evidence_ids.add(eid)
            evidence_lookup[eid] = item.get("content", "")

        has_usable_evidence = len(layer2_evidence_package) > 0 and any(item.get("content", "").strip() for item in layer2_evidence_package)

        retry_count = 0
        retry_instruction = None

        while retry_count <= 1:
            # 1. Generation Pass
            contract: GenerationContract = self.generator.generate(
                question=question,
                evidence_package=layer2_evidence_package,
                retry_instruction=retry_instruction
            )

            # 2. Fast-Fail Safety & Schema Checks
            schema_res: SafetyCheckResult = self.schema_validator.validate(contract, allowed_evidence_ids)
            if not schema_res.passed:
                action, reason = self.failure_router.route_failure(
                    safety_res=schema_res,
                    relevance_res=RelevanceResult(passed=True, score=1.0),
                    claim_results=[],
                    retry_count=retry_count,
                    has_usable_evidence=has_usable_evidence
                )
                if action == "RETRY" and retry_count < 1:
                    retry_count += 1
                    retry_instruction = reason
                    continue
                else:
                    return self._build_reject_decision(
                        query_id=query_id,
                        reason=reason,
                        retry_count=retry_count,
                        start_time=start_time
                    )

            safety_res: SafetyCheckResult = self.safety_scanner.scan(contract)
            if not safety_res.passed:
                action, reason = self.failure_router.route_failure(
                    safety_res=safety_res,
                    relevance_res=RelevanceResult(passed=True, score=1.0),
                    claim_results=[],
                    retry_count=retry_count,
                    has_usable_evidence=has_usable_evidence
                )
                if action == "RETRY" and retry_count < 1:
                    retry_count += 1
                    retry_instruction = reason
                    continue
                else:
                    return self._build_reject_decision(
                        query_id=query_id,
                        reason=reason,
                        retry_count=retry_count,
                        start_time=start_time
                    )

            # 3. Parallel Relevance & Direct Claim Verification
            relevance_res, claim_results = await self._run_parallel_verification(
                question=question,
                contract=contract,
                evidence_lookup=evidence_lookup
            )

            # 4. Failure-Type Router
            action, reason = self.failure_router.route_failure(
                safety_res=safety_res,
                relevance_res=relevance_res,
                claim_results=claim_results,
                retry_count=retry_count,
                has_usable_evidence=has_usable_evidence
            )

            if action == "RETRY" and retry_count < 1:
                retry_count += 1
                retry_instruction = reason
                continue

            if action == "REJECT":
                return self._build_reject_decision(
                    query_id=query_id,
                    reason=reason,
                    retry_count=retry_count,
                    start_time=start_time
                )

            # 5. Verified Answer Reconstruction
            reconstructed_answer, verified_claims, failed_claims = self.reconstructor.reconstruct(
                claims=contract.claims,
                verification_results=claim_results
            )

            # 6. Policy Enforcement
            policy_ok, policy_reason = self.policy_enforcer.evaluate_policy(
                policy_class=policy_class,
                verified_claims=verified_claims,
                failed_claims=failed_claims,
                verification_results=claim_results
            )

            if not policy_ok:
                return self._build_reject_decision(
                    query_id=query_id,
                    reason=policy_reason,
                    retry_count=retry_count,
                    start_time=start_time
                )

            # 7. Assemble Auditable PASS Decision
            exec_time = (time.time() - start_time) * 1000.0
            telemetry = self.telemetry_store.build_telemetry(
                query_id=query_id,
                model_version=self.model_version,
                decision="PASS",
                verified_claims=verified_claims,
                failed_claims=failed_claims,
                failure_reason=None,
                execution_time_ms=exec_time,
                claim_results=[r.model_dump() for r in claim_results],
                retry_count=retry_count,
            )

            return Layer3GateDecision(
                decision="PASS",
                reconstructed_answer=reconstructed_answer,
                verified_claims=verified_claims,
                failed_claims=failed_claims,
                retry_count=retry_count,
                telemetry=telemetry
            )

        return self._build_reject_decision(
            query_id=query_id,
            reason="Exhausted maximum retry attempts without verification pass.",
            retry_count=retry_count,
            start_time=start_time
        )

    async def _run_parallel_verification(
        self,
        question: str,
        contract: GenerationContract,
        evidence_lookup: Dict[str, str]
    ) -> tuple[RelevanceResult, List[ClaimVerificationResult]]:
        """Executes relevance check and direct claim NLI in parallel."""
        loop = asyncio.get_running_loop()

        relevance_task = loop.run_in_executor(
            None, self.relevance_evaluator.evaluate, question, contract
        )
        claim_task = loop.run_in_executor(
            None, self.claim_verifier.verify_claims, contract.claims, evidence_lookup
        )

        return await asyncio.gather(relevance_task, claim_task)

    def _build_reject_decision(
        self,
        query_id: str,
        reason: str,
        retry_count: int,
        start_time: float
    ) -> Layer3GateDecision:
        """Helper to construct REJECT decision with telemetry."""
        exec_time = (time.time() - start_time) * 1000.0
        telemetry = self.telemetry_store.build_telemetry(
            query_id=query_id,
            model_version=self.model_version,
            decision="REJECT",
            verified_claims=[],
            failed_claims=[],
            failure_reason=reason,
            execution_time_ms=exec_time,
            retry_count=retry_count,
        )

        return Layer3GateDecision(
            decision="REJECT",
            failure_reason=reason,
            retry_count=retry_count,
            telemetry=telemetry
        )
