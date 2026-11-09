"""Verifier v3: the extractive shortcut no longer joins words from different sentences (subject swaps)."""
import pytest

from app.rag.layer3.claim_verify import DirectClaimVerifier
from app.rag.layer3.contracts import ClaimObject

EVIDENCE = {
    "E1": "Employees must rotate passwords every 90 days. Contractors may request an MFA exception.",
    "E2": "Atlas is committed to 99.9% monthly uptime. If uptime falls below 99.9%, customers receive a service credit of "
          "10% of that month's fee. If uptime falls below 99.0%, the credit is 25%.",
}


@pytest.fixture(scope="module")
def verifier():
    v = DirectClaimVerifier()
    if not v.extractive_single_sentence:
        pytest.skip("verifier v2 (ATTS_L3_V3=0)")
    return v


def status(v, text):
    return v.verify_claims([ClaimObject(claim_id="c", order=0, text=text, evidence_ids=[])], EVIDENCE)[0].status


def test_subject_swap_across_sentences_is_not_released(verifier):
    assert status(verifier, "Employees can request an MFA exception.") != "ENTAILMENT"


def test_correct_subject_is_released(verifier):
    assert status(verifier, "Contractors may request an MFA exception.") == "ENTAILMENT"


def test_fact_spread_over_two_sentences_with_its_numbers_in_one_is_released(verifier):
    assert status(verifier, "Customers receive a service credit of 25% if monthly uptime falls below 99.0%.") == "ENTAILMENT"


@pytest.mark.parametrize("text", ["Customers receive a service credit of 25% if monthly uptime falls below 99.9%.",
                                  "Customers receive a service credit of 10% if monthly uptime falls below 99.0%."])
def test_numbers_mixed_across_sentences_are_not_released(verifier, text):
    assert status(verifier, text) != "ENTAILMENT"
