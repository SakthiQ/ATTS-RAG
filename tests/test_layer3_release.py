"""Layer 3 strict release: text that is not verified (headings, lead-ins, notes that the documents do not cover something,
the intent line and the chart's title, axis labels and unit) must not be able to state a fact of its own."""
import pytest

from app.rag.layer3 import generation
from app.rag.layer3.claim_verify import DirectClaimVerifier
from app.rag.layer3.contracts import ChartPoint, ChartSpec
from app.rag.layer3.gate import Layer3Gate
from app.rag.layer3.generation import NEUTRAL_CHART_TITLE, verify_chart

REFUNDS = "Refunds are accepted within 30 days of purchase. Store credit is offered after that."
QUESTION = "How long do I have to get a refund?"


@pytest.fixture(scope="module")
def verifier():
    return DirectClaimVerifier()


def ask(verifier, answer, intent="The user wants the refund window.", question=QUESTION, text=REFUNDS):
    gate = Layer3Gate(llm_invoker=lambda prompt: f"INTENT: {intent}\nANSWER:\n{answer}", verifier=verifier)
    package = [{"content": text, "metadata": {"chunk_uid": "u1", "source": "refunds.md", "tenant_id": "acme"}}]
    return gate.process(question, package, tenant_id="acme")


SUPPORTED = "Refunds are accepted within 30 days of purchase. [E1]"


# ---- unverified lines ----

@pytest.mark.parametrize("line", [
    "## Refunds take 90 days for managers",                                          # heading with a number
    "## Managers are exempt from receipts",                                          # heading with new words
    "Limit: 500 dollars.",                                                           # short line with a number
    "Managers get a 90-day window and a 2,000 dollar cap:",                          # lead-in with numbers
    "Managers are not required to provide a receipt for refunds up to 2,000 dollars.",   # a negated claim, uncited
    "The handbook does not provide further details, but it implies that managers can refund any amount.",
    "The evidence does not provide information beyond 30 days, so it is not possible to get a refund after 45 days.",
])
def test_an_unverified_line_cannot_state_a_fact(verifier, line):
    answer = f"{line}\n{SUPPORTED}" if line.startswith("#") else f"{SUPPORTED}\n{line}"
    res = ask(verifier, answer)
    assert res.decision == "PASS"
    assert res.reconstructed_answer == "Refunds are accepted within 30 days of purchase. [E1]"


def test_a_plain_note_that_the_documents_do_not_cover_something_is_kept(verifier):
    res = ask(verifier, f"{SUPPORTED}\nThe documents do not mention a refund limit for managers.")
    assert "The documents do not mention a refund limit for managers." in res.reconstructed_answer


def test_headings_and_lead_ins_that_only_repeat_the_answer_are_kept(verifier):
    res = ask(verifier, f"## Refunds\nHere are the key points:\n{SUPPORTED}")
    assert res.reconstructed_answer.startswith("## Refunds")
    assert "Here are the key points:" in res.reconstructed_answer


def test_a_short_line_with_a_number_is_verified_like_any_claim(verifier):
    res = ask(verifier, f"{SUPPORTED}\nLimit: 500 dollars.")
    assert "Limit: 500 dollars." in [c.text for c in res.failed_claims]
    assert res.grounding_ratio == 0.5


# ---- the intent line ----

def test_the_intent_is_one_sentence_without_new_numbers(verifier):
    res = ask(verifier, SUPPORTED, intent="The user wants the refund window. A numerical answer would help.")
    assert res.intent == "The user wants the refund window."
    assert ask(verifier, SUPPORTED, intent="The user wants to know that refunds take 90 days.").intent is None
    assert ask(verifier, SUPPORTED, intent="The user cannot get a refund.").intent is None


def test_a_secret_in_the_intent_rejects_the_answer(verifier):
    res = ask(verifier, SUPPORTED, intent="Refund window. Key: sk-live_9f8e7d6c5b4a3f2e1d0c9b8a")
    assert res.decision == "REJECT" and res.safety_flagged


# ---- charts ----

SHARES = "| Region | Share |\n| Q1 | 40% |\n| Q10 | 60% |\n| North | 25% |\n| South | 75% |"


def chart(title="Share by region", unit="%", y_label="Share", points=(("North", 25), ("South", 75))):
    return ChartSpec(type="bar", title=title, unit=unit, y_label=y_label,
                     data=[ChartPoint(label=l, value=v) for l, v in points])


def test_a_chart_with_evidence_backed_text_is_released_as_is():
    spec = chart()
    assert verify_chart(spec, {"E1": SHARES}, "Compare the regions") is spec


def test_a_chart_title_cannot_claim_what_the_data_does_not_show():
    out = verify_chart(chart(title="Sales collapsed by 90% this year"), {"E1": SHARES}, "Compare the regions")
    assert out.title == NEUTRAL_CHART_TITLE and [p.value for p in out.data] == [25, 75]
    out = verify_chart(chart(y_label="Not growing"), {"E1": SHARES}, "Compare the regions")
    assert out.y_label == ""


def test_a_unit_the_evidence_does_not_use_is_left_out():
    assert verify_chart(chart(unit="$"), {"E1": SHARES}, "Compare the regions").unit == ""


def test_chart_labels_match_whole_words_only():
    text = "| Quarter | Share |\n| Q10 | 40% |\n| Q2 | 60% |"
    assert verify_chart(chart(points=(("Q1", 40), ("Q2", 60))), {"E1": text}) is None


# ---- the switch that reproduces the earlier reports ----

def test_strict_release_can_be_switched_off_to_reproduce_earlier_reports(verifier, monkeypatch):
    monkeypatch.setattr(generation, "STRICT_RELEASE", False)
    negated = "Managers are not required to provide a receipt for refunds up to 2,000 dollars."
    res = ask(verifier, f"{SUPPORTED}\n{negated}",
              intent="The user wants the refund window. A numerical answer would help.")
    assert negated in res.reconstructed_answer
    assert res.intent == "The user wants the refund window. A numerical answer would help."
