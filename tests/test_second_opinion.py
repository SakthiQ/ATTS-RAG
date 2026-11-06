import json

import pytest

from app.rag.layer3.claim_verify import DirectClaimVerifier
from app.rag.layer3.contracts import ClaimObject
from app.rag.layer3.second_opinion import QuoteGroundedJudge

PTO = ("Paid time off accrues according to length of service.\n"
       "| Years of service | Annual PTO |\n| --- | --- |\n"
       "| 0 to 2 years | 15 days |\n| 3 to 5 years | 20 days |\n| 6 or more years | 25 days |")
CLAIM = "An employee with 4 years of service receives 20 days of annual PTO."


@pytest.fixture(scope="module")
def base():
    return DirectClaimVerifier()


def verifier_with(base, reply):
    import copy
    v = copy.copy(base)
    v.second_opinion = QuoteGroundedJudge(lambda prompt: reply if isinstance(reply, str) else json.dumps(reply))
    return v


def run(v, text=CLAIM, evidence=PTO):
    return v.verify_claims([ClaimObject(claim_id="C1", order=1, text=text, evidence_ids=["E1"])], {"E1": evidence})[0]


GOOD = {"verdict": "supported", "excerpt": "E1", "quote": "3 to 5 years | 20 days"}


def test_without_a_second_opinion_the_range_claim_is_left_insufficient(base):
    import copy
    plain = copy.copy(base)
    plain.second_opinion = None
    assert run(plain).status == "INSUFFICIENT"


def test_a_real_quote_that_passes_the_checks_releases_the_claim(base):
    res = run(verifier_with(base, GOOD))
    assert res.status == "ENTAILMENT" and "Second opinion" in res.reason and res.supporting_evidence_id == "E1"


def test_a_quote_that_is_not_in_the_evidence_is_rejected(base):
    fake = dict(GOOD, quote="4 years of service earn 20 days of leave")
    assert run(verifier_with(base, fake)).status == "INSUFFICIENT"


def test_a_quote_without_the_claims_number_is_rejected(base):
    wrong_row = dict(GOOD, quote="0 to 2 years | 15 days")   # real words, but 20 and 4 are not in it
    assert run(verifier_with(base, wrong_row)).status == "INSUFFICIENT"


def test_negation_mismatch_is_rejected(base):
    text = "Contractors may not request an MFA exception from the Security Office."
    evidence = "Contractors may request an MFA exception from the Security Office, but an exception expires after 30 days."
    reply = {"verdict": "supported", "excerpt": "E1", "quote": "Contractors may request an MFA exception from the Security Office"}
    assert run(verifier_with(base, reply), text, evidence).status != "ENTAILMENT"


def test_not_supported_and_garbage_replies_change_nothing(base):
    assert run(verifier_with(base, {"verdict": "not_supported"})).status == "INSUFFICIENT"
    assert run(verifier_with(base, "I think so, yes")).status == "INSUFFICIENT"


def test_reasons_and_opinions_are_never_sent_to_the_second_model(base):
    v = verifier_with(base, GOOD)
    opinion = "This policy exists to protect the company from reputational damage and to keep staff motivated."
    assert run(v, opinion).status != "ENTAILMENT"
    assert v.second_opinion.calls == 0
