import pytest
import asyncio
from unittest.mock import MagicMock
from app.rag.layer3.gate import Layer3Gate
from app.rag.layer3.contracts import Layer3GateDecision, GenerationContract, ClaimObject
from app.rag.layer3.safety import FastFailSafetyScanner
from app.rag.layer3.schema import SchemaValidator


def test_1_valid_claims_pass_and_reconstruct():
    """Test 1: Clean generation with valid cited evidence passes and reconstructs formatted answer."""
    gate = Layer3Gate()
    evidence_package = [
        {
            "content": "Official IT Policy: Employees must rotate passwords every 90 days.",
            "metadata": {"chunk_uid": "chk_101"}
        }
    ]

    res: Layer3GateDecision = asyncio.run(gate.process(
        question="What is the password policy?",
        layer2_evidence_package=evidence_package
    ))

    assert res.decision == "PASS"
    assert res.reconstructed_answer is not None
    assert "passwords" in res.reconstructed_answer.lower()
    assert len(res.verified_claims) == 1
    assert res.telemetry["decision"] == "PASS"


def test_2_claim_direct_contradiction_retry():
    """Test 2: Claim that directly contradicts evidence triggers 1-shot retry and succeeds on correct regeneration."""
    call_count = 0

    def contradiction_invoker(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return '{"claims": [{"claim_id": "C1", "order": 1, "text": "Passwords never expire.", "evidence_ids": ["E1"]}]}'
        else:
            return '{"claims": [{"claim_id": "C1", "order": 1, "text": "Passwords expire every 90 days.", "evidence_ids": ["E1"]}]}'

    gate = Layer3Gate(llm_invoker=contradiction_invoker)
    evidence_package = [{"content": "Official policy: Passwords expire every 90 days.", "metadata": {"chunk_uid": "E1"}}]

    res: Layer3GateDecision = asyncio.run(gate.process(
        question="When do passwords expire?",
        layer2_evidence_package=evidence_package
    ))

    assert res.decision == "PASS"
    assert res.retry_count == 1


def test_3_insufficient_claim_retry_and_recovery():
    """Test 3: Over-specified claim resulting in INSUFFICIENT with usable evidence available triggers 1-shot retry."""
    call_count = 0

    def overspecified_invoker(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Over-specified claim: evidence says periodically, claim specifies every 30 days
            return '{"claims": [{"claim_id": "C1", "order": 1, "text": "Passwords must be changed every 30 days.", "evidence_ids": ["E1"]}]}'
        else:
            return '{"claims": [{"claim_id": "C1", "order": 1, "text": "Passwords must be changed periodically.", "evidence_ids": ["E1"]}]}'

    gate = Layer3Gate(llm_invoker=overspecified_invoker)
    evidence_package = [{"content": "Passwords must be changed periodically.", "metadata": {"chunk_uid": "E1"}}]

    res: Layer3GateDecision = asyncio.run(gate.process(
        question="What is the password policy?",
        layer2_evidence_package=evidence_package
    ))

    assert res.decision == "PASS"
    assert res.retry_count == 1
    assert "periodically" in res.reconstructed_answer


def test_4_missing_evidence_id_contract_retry():
    """Test 4: Claim missing evidence ID when evidence exists in Layer 2 package triggers contract retry."""
    call_count = 0

    def missing_id_invoker(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return '{"claims": [{"claim_id": "C1", "order": 1, "text": "Employees must rotate passwords every 90 days.", "evidence_ids": []}]}'
        else:
            return '{"claims": [{"claim_id": "C1", "order": 1, "text": "Employees must rotate passwords every 90 days.", "evidence_ids": ["chk_101"]}]}'

    gate = Layer3Gate(llm_invoker=missing_id_invoker)
    evidence_package = [{"content": "Employees must rotate passwords every 90 days.", "metadata": {"chunk_uid": "chk_101"}}]

    res: Layer3GateDecision = asyncio.run(gate.process(
        question="What is the password policy?",
        layer2_evidence_package=evidence_package
    ))

    assert res.decision == "PASS"
    assert res.retry_count == 1


def test_5_invalid_evidence_id_allowlist_retry():
    """Test 5: Citing an evidence ID outside the Layer 2 allow-list triggers 1-shot retry."""
    call_count = 0

    def retry_invoker(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return '{"claims": [{"claim_id": "C1", "order": 1, "text": "Employees must rotate passwords.", "evidence_ids": ["E_fake"]}]}'
        else:
            return '{"claims": [{"claim_id": "C1", "order": 1, "text": "Employees must rotate passwords every 90 days.", "evidence_ids": ["chk_valid"]}]}'

    gate = Layer3Gate(llm_invoker=retry_invoker)
    evidence_package = [{"content": "Employees must rotate passwords every 90 days.", "metadata": {"chunk_uid": "chk_valid"}}]

    res: Layer3GateDecision = asyncio.run(gate.process(
        question="What is the password policy?",
        layer2_evidence_package=evidence_package
    ))

    assert res.decision == "PASS"
    assert res.retry_count == 1


def test_6_evidence_gap_immediate_reject():
    """Test 6: Unsupported claim with no usable supporting evidence in Layer 2 is classified as evidence gap and immediately REJECTED."""
    def unsupported_invoker(prompt: str) -> str:
        return '{"claims": [{"claim_id": "C1", "order": 1, "text": "Company provides 100% paid remote retreats in Hawaii.", "evidence_ids": []}]}'

    gate = Layer3Gate(llm_invoker=unsupported_invoker)
    # Package has no usable text / empty package
    evidence_package = []

    res: Layer3GateDecision = asyncio.run(gate.process(
        question="What are the remote retreat policies?",
        layer2_evidence_package=evidence_package
    ))

    assert res.decision == "REJECT"
    assert res.retry_count == 0  # Zero wasted retries for evidence gaps
    assert "evidence gap" in res.failure_reason.lower() or "evidence" in res.failure_reason.lower()


def test_7_security_critical_policy_failure():
    """Test 7: Request under SECURITY_CRITICAL policy missing required assertions causes REJECT."""
    gate = Layer3Gate()

    # Pass empty evidence package so zero claims verify
    res: Layer3GateDecision = asyncio.run(gate.process(
        question="What is the security policy?",
        layer2_evidence_package=[],
        policy_class="SECURITY_CRITICAL"
    ))

    assert res.decision == "REJECT"


def test_8_secret_leak_fast_fail_reject():
    """Test 8: Output containing secret key triggers immediate fast-fail REJECT and asserts parallel verifiers did NOT run."""
    def leaking_invoker(prompt: str) -> str:
        return '{"claims": [{"claim_id": "C1", "order": 1, "text": "The API key is sk-proj1234567890abcdef123456", "evidence_ids": ["E1"]}]}'

    gate = Layer3Gate(llm_invoker=leaking_invoker)

    # Mock parallel verifiers to assert they were NOT called
    gate._run_parallel_verification = MagicMock()

    evidence_package = [{"content": "Dummy text.", "metadata": {"chunk_uid": "E1"}}]

    res: Layer3GateDecision = asyncio.run(gate.process(
        question="What is the key?",
        layer2_evidence_package=evidence_package
    ))

    assert res.decision == "REJECT"
    assert "secret" in res.failure_reason.lower()
    gate._run_parallel_verification.assert_not_called()  # Fast-fail execution assertion


def test_9_cross_tenant_leak_reject():
    """Test 9: Cross-tenant metadata mismatch triggers immediate REJECT."""
    gate = Layer3Gate()
    evidence_package = [{"content": "Confidential data.", "metadata": {"chunk_uid": "E1", "tenant_id": "tenant-B"}}]

    res: Layer3GateDecision = asyncio.run(gate.process(
        question="What is the confidential data?",
        layer2_evidence_package=evidence_package,
        tenant_id="tenant-A"
    ))

    assert res.decision == "REJECT"
    assert "cross-tenant" in res.failure_reason.lower()


def test_10_unverified_prose_reconstruction_isolation():
    """Test 10: Verified reconstruction strictly isolates output to verified claims and excludes unverified prose."""
    call_count = 0

    def multi_claim_invoker(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return '''{
                "claims": [
                    {"claim_id": "C1", "order": 1, "text": "Official password policy requires employees to rotate passwords every 90 days.", "evidence_ids": ["E1"]},
                    {"claim_id": "C2", "order": 2, "text": "Passwords never expire.", "evidence_ids": ["E1"]}
                ]
            }'''
        else:
            return '''{
                "claims": [
                    {"claim_id": "C1", "order": 1, "text": "Official password policy requires employees to rotate passwords every 90 days.", "evidence_ids": ["E1"]}
                ]
            }'''

    gate = Layer3Gate(llm_invoker=multi_claim_invoker)
    evidence_package = [{"content": "Official password policy requires employees to rotate passwords every 90 days.", "metadata": {"chunk_uid": "E1"}}]

    res: Layer3GateDecision = asyncio.run(gate.process(
        question="What is the password policy?",
        layer2_evidence_package=evidence_package,
        policy_class="ORDINARY_INFORMATIONAL"
    ))

    assert res.decision == "PASS"
    assert len(res.verified_claims) == 1
    assert res.verified_claims[0].claim_id == "C1"
    assert "never expire" not in res.reconstructed_answer


def test_11_verifier_failure_unknown_fail_closed():
    """Test 11: Verifier failure yielding UNKNOWN status fails closed to REJECT."""
    gate = Layer3Gate()

    # Ensure relevance check passes to isolate verifier failure
    from app.rag.layer3.contracts import RelevanceResult, ClaimVerificationResult
    gate.relevance_evaluator.evaluate = lambda q, c: RelevanceResult(passed=True, score=1.0)

    # Force claim verifier to return UNKNOWN status
    def failing_verifier(claims, lookup):
        return [
            ClaimVerificationResult(
                claim_id="C1",
                text="Some claim text.",
                cited_evidence_ids=["E1"],
                status="UNKNOWN",
                reason="Verifier service timeout"
            )
        ]

    gate.claim_verifier.verify_claims = failing_verifier
    evidence_package = [{"content": "Some claim text.", "metadata": {"chunk_uid": "E1"}}]

    res: Layer3GateDecision = asyncio.run(gate.process(
        question="What is the policy?",
        layer2_evidence_package=evidence_package
    ))

    assert res.decision == "REJECT"
    assert "UNKNOWN" in res.failure_reason


def test_12_exhausted_retry_rejection():
    """Test 12: Retry output that fails verification a second time is immediately REJECTED."""
    def persistent_bad_invoker(prompt: str) -> str:
        return '{"claims": [{"claim_id": "C1", "order": 1, "text": "Passwords never expire.", "evidence_ids": ["E1"]}]}'

    gate = Layer3Gate(llm_invoker=persistent_bad_invoker)
    evidence_package = [{"content": "Official policy: Passwords expire every 90 days.", "metadata": {"chunk_uid": "E1"}}]

    res: Layer3GateDecision = asyncio.run(gate.process(
        question="When do passwords expire?",
        layer2_evidence_package=evidence_package
    ))

    assert res.decision == "REJECT"
    assert res.retry_count == 1




