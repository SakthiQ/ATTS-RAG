import time
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from app.rag.layer3.claim_verify import DirectClaimVerifier
from app.rag.layer3.contracts import ClaimObject, Layer3GateDecision
from app.rag.layer3.generation import AnswerGenerator, release_intent, wants_chart
from app.rag.layer3.reconstruction import VerifiedAnswerReconstructor
from app.rag.layer3.safety import FastFailSafetyScanner
from app.rag.layer3.telemetry import Layer3TelemetryStore


class Layer3Gate:
    """Layer 3 - Evidence-to-Answer & Output Verification Gate.

    Pipeline: draft a descriptive cited answer -> secret/prompt-leak scan -> sentence-level NLI
    verification against the evidence -> rebuild the answer from verified sentences only ->
    (optional) evidence-checked chart. One LLM call for the answer, one more only if a chart is plausible.
    """

    def __init__(
        self,
        llm_invoker: Optional[Callable[[str], str]] = None,
        model_version: str = "generator-v2",
        verifier: Optional[DirectClaimVerifier] = None,
        json_invoker: Optional[Callable[[str], str]] = None,
    ):
        self.generator = AnswerGenerator(llm_invoker=llm_invoker, json_invoker=json_invoker)
        self.safety_scanner = FastFailSafetyScanner()
        self.claim_verifier = verifier or DirectClaimVerifier()
        self.reconstructor = VerifiedAnswerReconstructor()
        self.telemetry_store = Layer3TelemetryStore()
        self.model_version = model_version

    def process(
        self,
        question: str,
        layer2_evidence_package: List[Dict[str, Any]],
        query_id: str = "Q-101",
        tenant_id: Optional[str] = None,
        policy: Optional[Any] = None,
    ) -> Layer3GateDecision:
        """`policy` (an AdaptivePolicy) tightens verification for a suspicious user: a higher entailment
        threshold, and a minimum share of the draft that must verify."""
        start = time.time()
        extra = {"policy": policy.as_dict()} if policy is not None else None
        threshold = getattr(policy, "entailment_threshold", None)

        # Defense in depth: Layer 2 already filters by tenant, so a mismatch here means a bug upstream
        if tenant_id:
            for item in layer2_evidence_package:
                item_tenant = item.get("metadata", {}).get("tenant_id")
                if item_tenant and item_tenant != tenant_id:
                    return self._reject(query_id, start,
                                        f"Cross-tenant authorization violation: evidence tenant '{item_tenant}' "
                                        f"does not match query tenant '{tenant_id}'.")

        t0 = time.time()
        draft = self.generator.generate(question, layer2_evidence_package)
        t_generate = time.time() - t0
        if not draft.evidence_lookup:
            return self._reject(query_id, start, "Evidence gap: no usable evidence in the Layer 2 package.")
        if not draft.claims:
            return self._reject(query_id, start, "The model returned no checkable answer.")

        # The intent line is shown to the user too, so it is scanned with the answer
        draft_text = " ".join([draft.intent or ""] + [s.text for line in draft.lines for s in line.sentences])
        safety = self.safety_scanner.scan(draft_text)
        if not safety.passed:
            return self._reject(query_id, start, safety.reason or "Unsafe output detected.", safety_flagged=True, extra=extra)

        t0 = time.time()
        results = self.claim_verifier.verify_claims(draft.claims, draft.evidence_lookup, entailment_threshold=threshold)
        contradictions = sum(1 for r in results if r.status == "CONTRADICTION")
        t_verify = time.time() - t0
        answer, verified, failed = self.reconstructor.reconstruct(draft, results)

        if not verified:
            statuses = sorted({r.status for r in results})
            return self._reject(query_id, start,
                                f"No sentence of the draft answer could be verified against the evidence ({', '.join(statuses)}).",
                                claim_results=results, failed=failed, contradictions=contradictions, extra=extra)

        ratio = len(verified) / max(len(draft.claims), 1)
        min_grounding = getattr(policy, "min_grounding", 0.0)
        if min_grounding and ratio < min_grounding:
            return self._reject(query_id, start,
                                f"Only {ratio:.0%} of the draft could be verified; {min_grounding:.0%} is required while the "
                                "session is under suspicion.",
                                claim_results=results, failed=failed, contradictions=contradictions, extra=extra)

        chart = None
        t0 = time.time()
        if wants_chart(question, draft.evidence_lookup):
            try:
                chart = self.generator.propose_chart(question, draft.evidence_lookup)
                if chart is not None and not self.safety_scanner.scan(
                        " ".join([chart.title, chart.x_label, chart.y_label, chart.unit] + [p.label for p in chart.data])).passed:
                    chart = None
            except Exception as exc:  # a chart is optional; never lose a verified answer over it
                logger.warning(f"Chart generation failed: {exc}")

        t_chart = time.time() - t0
        logger.info(f"Layer 3 timings: generate={t_generate:.1f}s verify={t_verify:.1f}s chart={t_chart:.1f}s")
        telemetry = self.telemetry_store.build_telemetry(
            query_id=query_id, model_version=self.model_version, decision="PASS",
            verified_claims=verified, failed_claims=failed, failure_reason=None,
            execution_time_ms=(time.time() - start) * 1000.0,
            claim_results=[r.model_dump() for r in results], extra=extra,
        )
        return Layer3GateDecision(
            decision="PASS",
            reconstructed_answer=answer,
            intent=release_intent(draft.intent, question),
            chart=chart,
            verified_claims=verified,
            failed_claims=failed,
            grounding_ratio=round(ratio, 3),
            evidence_map=draft.evidence_to_uid,
            telemetry=telemetry,
            contradictions=contradictions,
        )

    def _reject(self, query_id: str, start: float, reason: str,
                claim_results: Optional[list] = None, failed: Optional[List[ClaimObject]] = None,
                safety_flagged: bool = False, contradictions: int = 0, extra: Optional[dict] = None) -> Layer3GateDecision:
        telemetry = self.telemetry_store.build_telemetry(
            query_id=query_id, model_version=self.model_version, decision="REJECT",
            verified_claims=[], failed_claims=failed or [], failure_reason=reason,
            execution_time_ms=(time.time() - start) * 1000.0,
            claim_results=[r.model_dump() for r in claim_results] if claim_results else None, extra=extra,
        )
        return Layer3GateDecision(decision="REJECT", failure_reason=reason, telemetry=telemetry,
                                  failed_claims=failed or [], safety_flagged=safety_flagged, contradictions=contradictions)
