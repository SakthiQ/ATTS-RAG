import pytest

from app.rag.layer3.claim_verify import DirectClaimVerifier
from app.rag.layer3.contracts import ChartPoint, ChartSpec, ClaimObject
from app.rag.layer3.gate import Layer3Gate
from app.rag.layer3.generation import AnswerGenerator, verify_chart, wants_chart
from app.rag.layer3.safety import FastFailSafetyScanner

UID = "8da1d6b48a0a49571f25366a0115f0e74b54d9bd794ff410f38cd784db2eb8d8_50"  # real-world chunk ids are long hashes

POLICY = (
    "Official IT Policy: Employees must rotate passwords every 90 days. "
    "Multi-factor authentication is required for all remote access. "
    "Contractors may request an exception from the security office."
)


@pytest.fixture(scope="module")
def verifier():
    return DirectClaimVerifier()


def evidence(text=POLICY, uid=UID, **meta):
    return [{"content": text, "metadata": {"chunk_uid": uid, "source": "policy.pdf", "page": 3, **meta}}]


def gate_with(verifier, reply):
    return Layer3Gate(llm_invoker=lambda prompt: reply, verifier=verifier)


# --------------------------------------------------------------------------- end-to-end gate

def test_verified_answer_is_released_with_intent_and_short_citations(verifier):
    reply = (
        "INTENT: The user wants the password rotation rule.\n"
        "ANSWER:\nEmployees must rotate passwords every 90 days. [E1]\n"
        "Multi-factor authentication is required for all remote access. [E1]"
    )
    res = gate_with(verifier, reply).process("What is the password policy?", evidence())

    assert res.decision == "PASS"
    assert res.intent == "The user wants the password rotation rule."
    assert len(res.verified_claims) == 2 and not res.failed_claims
    assert "rotate passwords every 90 days" in res.reconstructed_answer
    assert "[E1]" in res.reconstructed_answer
    assert res.evidence_map == {"E1": UID}  # the LLM never sees the long hash
    assert res.grounding_ratio == 1.0


def test_unsupported_and_contradicted_sentences_are_dropped_but_grounded_ones_survive(verifier):
    reply = (
        "INTENT: password rules\nANSWER:\n"
        "Employees must rotate passwords every 90 days. [E1]\n"
        "Passwords never expire. [E1]\n"
        "Employees must rotate passwords every 30 days. [E1]"
    )
    res = gate_with(verifier, reply).process("When do passwords expire?", evidence())

    assert res.decision == "PASS"
    assert [c.text for c in res.verified_claims] == ["Employees must rotate passwords every 90 days."]
    assert len(res.failed_claims) == 2
    assert "never expire" not in res.reconstructed_answer
    assert "30 days" not in res.reconstructed_answer  # number not in the evidence
    assert res.grounding_ratio == pytest.approx(1 / 3, abs=0.01)


def test_answer_with_nothing_verifiable_is_rejected(verifier):
    reply = "INTENT: x\nANSWER:\nThe company was founded in 1985 by two engineers. [E1]"
    res = gate_with(verifier, reply).process("Who founded the company?", evidence())
    assert res.decision == "REJECT"
    assert res.reconstructed_answer is None


def test_wrong_citation_is_repaired_to_the_supporting_passage(verifier):
    package = evidence() + [{"content": "The cafeteria opens at 8am and closes at 4pm on weekdays.",
                             "metadata": {"chunk_uid": "uid_b", "source": "faq.pdf"}}]
    reply = "INTENT: x\nANSWER:\nEmployees must rotate passwords every 90 days. [E2]"
    res = gate_with(verifier, reply).process("Password rotation?", package)

    assert res.decision == "PASS"
    assert res.verified_claims[0].evidence_ids == ["E1"]
    assert "[E1]" in res.reconstructed_answer and "[E2]" not in res.reconstructed_answer


def test_secret_in_generated_text_is_rejected(verifier):
    reply = "INTENT: x\nANSWER:\nThe admin key is sk-abcdefghijklmnopqrstuvwxyz123456. [E1]"
    res = gate_with(verifier, reply).process("What is the key?", evidence())
    assert res.decision == "REJECT"
    assert "secret" in res.failure_reason.lower()


def test_cross_tenant_evidence_is_rejected(verifier):
    res = gate_with(verifier, "ANSWER: x").process("q", evidence(tenant_id="tenant_b"), tenant_id="tenant_a")
    assert res.decision == "REJECT"
    assert "cross-tenant" in res.failure_reason.lower()


def test_empty_llm_reply_is_rejected_not_crashed(verifier):
    res = gate_with(verifier, "").process("q", evidence())
    assert res.decision == "REJECT"


def test_telemetry_records_every_decision(verifier):
    reply = "INTENT: x\nANSWER:\nEmployees must rotate passwords every 90 days. [E1]"
    res = gate_with(verifier, reply).process("q", evidence())
    assert res.telemetry["decision"] == "PASS"
    assert res.telemetry["claim_results"][0]["status"] == "ENTAILMENT"


# --------------------------------------------------------------------------- verification details

def test_list_items_copied_from_evidence_pass_extractive_check(verifier):
    text = "Pitfall #4: Neglecting Quick Wins (50% of failures)\nPitfall #5: Underestimating Change Management (45% of failures)"
    claims = [
        ClaimObject(claim_id="C1", order=1, text="Neglecting Quick Wins (50% of failures)", evidence_ids=["E1"]),
        ClaimObject(claim_id="C2", order=2, text="Underestimating Change Management (65% of failures)", evidence_ids=["E1"]),
    ]
    results = verifier.verify_claims(claims, {"E1": text})
    assert results[0].status == "ENTAILMENT"
    assert results[1].status != "ENTAILMENT"  # 65 is not in the evidence


def test_numbers_must_appear_in_evidence(verifier):
    claim = ClaimObject(claim_id="C1", order=1, text="These pitfalls account for 135% of failures.", evidence_ids=["E1"])
    result = verifier.verify_claims([claim], {"E1": "Pitfall one causes 50% of failures."})[0]
    assert result.status == "INSUFFICIENT" and "135" in result.reason


# --------------------------------------------------------------------------- parsing

def test_parser_handles_markdown_labels_attribution_phrases_and_bullets():
    ids = {"E1", "E2"}
    raw = (
        "**INTENT**: Wants causes.\n\n**ANSWER**:\n"
        "According to [E1], 70% of failures come from pilots. This is bad. [E1]\n"
        "Additionally, [E2] highlights that quick wins matter.\n"
        "- First item [E1, E2].\n"
        "| a | b |\n"
        "Here is a chart of the results:"
    )
    intent, lines = AnswerGenerator.parse(raw, ids)
    assert intent == "Wants causes."
    texts = [(l.prefix, s.text, s.evidence_ids) for l in lines for s in l.sentences]
    assert ("", "70% of failures come from pilots.", ["E1"]) in texts
    assert ("", "Additionally, quick wins matter.", ["E2"]) in texts
    assert ("- ", "First item.", ["E1", "E2"]) in texts
    assert not any("chart" in t[1] or "|" in t[1] for t in texts)


def test_parser_uses_text_after_the_last_answer_label_and_drops_unknown_ids():
    raw = "INTENT: a\nANSWER:\nINTENT: a\nANSWER:\nFact one is true. [E9][E1]"
    intent, lines = AnswerGenerator.parse(raw, {"E1"})
    assert [s.evidence_ids for l in lines for s in l.sentences] == [["E1"]]


def test_evidence_labels_are_short_and_budgeted():
    package = [{"content": "x" * 5000, "metadata": {"chunk_uid": f"hash{i}"}} for i in range(3)]
    block, lookup, to_uid = AnswerGenerator.label_evidence(package, char_budget=7000)
    assert list(lookup) == ["E1"] or list(lookup) == ["E1", "E2"][:len(lookup)]
    assert all(k.startswith("E") and len(k) <= 3 for k in lookup)
    assert to_uid["E1"] == "hash0"


# --------------------------------------------------------------------------- charts

TABLE = "| Pitfall | Share |\n| Neglecting Quick Wins | 50% |\n| Wrong Use Case Selection | 40% |\n| Change Management | 45% |"


def _chart(*points):
    return ChartSpec(type="bar", title="t", data=[ChartPoint(label=l, value=v) for l, v in points])


def test_chart_with_traceable_data_is_accepted():
    spec = _chart(("Neglecting Quick Wins", 50), ("Change Management", 45))
    assert verify_chart(spec, {"E1": TABLE}) is spec


def test_chart_with_invented_value_or_label_is_rejected():
    assert verify_chart(_chart(("Neglecting Quick Wins", 55), ("Change Management", 45)), {"E1": TABLE}) is None
    assert verify_chart(_chart(("Budget Overruns", 50), ("Change Management", 45)), {"E1": TABLE}) is None
    assert verify_chart(_chart(("Neglecting Quick Wins", 50)), {"E1": TABLE}) is None  # needs two points


def test_chart_values_cannot_be_borrowed_across_passages():
    lookup = {"E1": "Quick Wins matter a lot.", "E2": "Revenue was 50 million."}
    assert verify_chart(_chart(("Quick Wins", 50), ("Revenue", 50)), lookup) is None


def test_wants_chart_needs_numbers_and_a_quantitative_question():
    assert wants_chart("Compare failure shares", {"E1": TABLE})
    assert not wants_chart("Who is the sponsor?", {"E1": TABLE})
    assert not wants_chart("Compare failure shares", {"E1": "No digits here at all."})


# --------------------------------------------------------------------------- safety scanner

def test_safety_scanner_flags_secrets_and_prompt_leakage():
    scanner = FastFailSafetyScanner()
    assert scanner.scan("Employees rotate passwords every 90 days.").passed
    assert scanner.scan("key AKIAIOSFODNN7EXAMPLE").leaked_secrets_found
    assert not scanner.scan("System Prompt: you are a helpful bot").passed


# --------------------------------------------------------------------------- tables

PIPE_TABLE = (
    "| Severity | Description | Target response time |\n| --- | --- | --- |\n"
    "| P1 | Active data breach or outage of a critical system | 15 minutes |\n"
    "| P4 | Low-risk request or question | 3 days |"
)


def test_pipe_table_rows_become_sentences():
    from app.rag.layer3.claim_verify import split_sentences
    sents = split_sentences(PIPE_TABLE)
    assert sents[0] == "Severity: P1; Description: Active data breach or outage of a critical system; Target response time: 15 minutes."
    assert len(sents) == 2


def test_claims_about_pipe_table_rows_are_judged_correctly(verifier):
    claims = [
        ClaimObject(claim_id="C1", order=1, text="P1 incidents have a target response time of 15 minutes.", evidence_ids=["E1"]),
        ClaimObject(claim_id="C2", order=2, text="P1 incidents have a target response time of 3 days.", evidence_ids=["E1"]),
    ]
    results = verifier.verify_claims(claims, {"E1": PIPE_TABLE})
    assert results[0].status == "ENTAILMENT"
    assert results[1].status != "ENTAILMENT"


# --------------------------------------------------------------------------- adaptive policy

class FakeVerifier:
    """Entails 'good...' claims at confidence 0.7 and 'great...' claims at 0.9; used to test the policy plumbing."""

    def verify_claims(self, claims, evidence_lookup, entailment_threshold=None):
        from app.rag.layer3.contracts import ClaimVerificationResult
        threshold = 0.6 if entailment_threshold is None else entailment_threshold
        out = []
        for c in claims:
            confidence = 0.9 if c.text.startswith("great") else 0.7 if c.text.startswith("good") else 0.0
            ok = confidence > 0 and confidence >= threshold
            out.append(ClaimVerificationResult(claim_id=c.claim_id, text=c.text, cited_evidence_ids=c.evidence_ids,
                                               status="ENTAILMENT" if ok else "INSUFFICIENT", confidence=confidence,
                                               supporting_evidence_id=c.evidence_ids[0] if ok else None))
        return out


def _fake_gate(reply):
    return Layer3Gate(llm_invoker=lambda prompt: reply, verifier=FakeVerifier())


def test_a_suspicious_user_faces_a_stricter_entailment_threshold():
    from app.rag.session_trust import policy_for
    reply = "INTENT: x\nANSWER:\ngood sentence one is here. [E1]"
    assert _fake_gate(reply).process("q", evidence()).decision == "PASS"
    assert _fake_gate(reply).process("q", evidence(), policy=policy_for(0.3)).decision == "PASS"   # threshold 0.675 < 0.7
    strict = _fake_gate(reply).process("q", evidence(), policy=policy_for(0.9))                    # threshold 0.825 > 0.7
    assert strict.decision == "REJECT"
    assert strict.telemetry["policy"]["entailment_threshold"] == pytest.approx(0.825)


def test_a_highly_suspicious_user_needs_half_of_the_draft_to_verify():
    from app.rag.session_trust import policy_for
    reply = ("INTENT: x\nANSWER:\ngreat first sentence stands here. [E1]\n"
             "bad second sentence is invented. [E1]\nbad third sentence is invented. [E1]")
    assert _fake_gate(reply).process("q", evidence()).decision == "PASS"                            # 1 of 3 verified is enough normally
    res = _fake_gate(reply).process("q", evidence(), policy=policy_for(0.8))
    assert res.decision == "REJECT" and "under suspicion" in res.failure_reason


def test_the_decision_reports_evidence_for_the_adaptive_layers(verifier):
    leak = gate_with(verifier, "INTENT: x\nANSWER:\nThe admin key is sk-abcdefghijklmnopqrstuvwxyz123456. [E1]").process("q", evidence())
    assert leak.safety_flagged is True
    table = evidence(PIPE_TABLE)
    contradicted = gate_with(verifier, "INTENT: x\nANSWER:\nP1 incidents have a target response time of 3 days. [E1]\n"
                                        "P1 incidents have a target response time of 15 minutes. [E1]").process("q", table)
    assert contradicted.contradictions == 1 and contradicted.safety_flagged is False
