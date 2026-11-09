"""Verifier v3 (reviewer point 4): the table-row check accepts correct table facts and nothing else."""
import pytest

from app.rag.layer3.table_rows import check_table_row_claim

PLANS = """# Acme Atlas Support Plans (2026)

Atlas support is offered in three plans, as shown below.

| Plan | Seats included | Monthly price per seat | First response time |
| --- | --- | --- | --- |
| Basic | 1 to 10 seats | $12 | 24 hours |
| Business | 11 to 100 seats | $9 | 4 hours |
| Enterprise | 101 or more seats | $7 | 30 minutes |

Annual billing gives a 15% discount on every plan. Customers receive a service credit if uptime falls short."""

COMMISSION = """## Commission rates

Sales representatives earn commission on closed deals according to the annual quota attainment tiers below.

| Quota attainment | Commission rate |
| --- | --- |
| Below 50% | 2% |
| 50% to 99% | 5% |
| 100% to 149% | 8% |
| 150% or more | 12% |"""

SEVERITY = """The Security Office classifies each incident and responds within the target time.

| Severity | Description | Target response time |
| --- | --- | --- |
| P1 | Active data breach or outage of a critical system | 15 minutes |
| P2 | Suspected breach or major degradation of service | 1 hour |
| P3 | Contained policy violation | 8 hours |
| P4 | Low-risk request or question | 3 days |"""

PTO = """Employees receive paid time off (PTO) each year, based on their length of service.

| Years of service | Annual PTO |
| --- | --- |
| 0 to 2 years | 15 days |
| 3 to 5 years | 20 days |
| 6 or more years | 25 days |"""

FINANCE = """Total Q3 revenue was $95.2 million, up 7.8% from Q2.

| Region | Revenue | Share of total |
| --- | --- | --- |
| North America | 42.1 | 44.2% |
| EMEA | 27.8 | 29.2% |
| APAC | 18.4 | 19.3% |
| LATAM | 6.9 | 7.2% |

| Quarter | Revenue |
| --- | --- |
| Q1 2026 | 81.5 |
| Q2 2026 | 88.3 |
| Q3 2026 | 95.2 |"""

CORRECT = [
    (PLANS, "The Business support plan costs $9 per seat per month."),
    (PLANS, "The Basic plan has a monthly price per seat of $12, while the Business plan has a price of $9 per seat, "
            "and the Enterprise plan has a price of $7 per seat."),
    (PLANS, "Business plan customers get a first response within 4 hours."),
    (COMMISSION, "A sales representative at 120% of quota earns a commission rate of 8%."),
    (COMMISSION, "The commission rates for different quota attainment tiers are 2%, 5%, 8%, and 12%."),
    (SEVERITY, "The target response times for P1 and P2 incidents are 15 minutes and 1 hour, respectively."),
    (SEVERITY, "The Security Office responds to P1 incidents, which are active data breaches or outages of critical "
               "systems, within 15 minutes."),
    (PTO, "An employee with 4 years of service receives 20 days of PTO each year."),
    (FINANCE, "The revenue in Q2 2026 was $88.3 million."),
    (FINANCE, "Q3 revenue by region was $95.2 million, with North America at $42.1 million (44.2% of total) followed "
              "by EMEA ($27.8 million, 29.2% of total)."),
]

WRONG = [
    (PLANS, "The Business support plan costs $12 per seat per month."),        # $12 is Basic, and 12 seats is not $12
    (PLANS, "The Enterprise plan costs $9 per seat per month."),               # wrong row
    (PLANS, "The Business plan has a first response time of 9 hours."),        # wrong column
    (PLANS, "The Business plan costs $9 per seat per month and includes free onboarding training."),   # added claim
    (PLANS, "The Business plan does not cost $9 per seat."),                   # negation
    (PLANS, "The plan that fits a company with 40 seats is the Basic plan."),  # 40 seats is Business
    (PLANS, "The Business plan includes 9 seats."),                            # $9 is not 9 seats
    (COMMISSION, "A sales representative at 120% of quota earns a commission rate of 12%."),
    (COMMISSION, "A sales representative at 160% of quota earns a commission rate of 8%."),
    (COMMISSION, "A sales representative at 40% of quota earns a commission rate of 5%."),
    (SEVERITY, "The target response times for P1 and P2 incidents are 1 hour and 15 minutes, respectively."),
    (SEVERITY, "P1 incidents must be answered within 15 hours."),               # unit
    (SEVERITY, "P3 incidents have a target response time of 3 days."),
    (PTO, "An employee with 4 years of service receives 25 days of PTO each year."),
    (PTO, "An employee with 1 year of service receives 20 days of PTO each year."),
    (PTO, "An employee with 7 years of service receives 20 days of PTO each year."),
    (FINANCE, "The revenue in Q1 2026 was $88.3 million."),
    (FINANCE, "EMEA revenue was $42.1 million."),
    (FINANCE, "North America accounted for 29.2% of total revenue."),
    (FINANCE, "Q3 revenue was $88.3 million, with North America at $42.1 million."),   # 88.3 is Q2
    (FINANCE, "The majority of Q3 revenue came from North America ($42.1 million, 44.2% of total)."),   # 44.2% is no majority
]


@pytest.mark.parametrize("passage,claim", CORRECT)
def test_correct_table_facts_are_accepted(passage, claim):
    assert check_table_row_claim(claim, {"E1": passage}) is not None


@pytest.mark.parametrize("passage,claim", WRONG)
def test_wrong_table_facts_are_not_accepted(passage, claim):
    assert check_table_row_claim(claim, {"E1": passage}) is None


def test_claims_without_numbers_are_left_to_nli():
    assert check_table_row_claim("The Business plan is the best choice.", {"E1": PLANS}) is None
