from app.rag.layer3.table_check import check_table_claim, parse_tables

REVENUE = """Acme > Revenue by region (Q3 2026, USD millions)
| Region | Revenue | Share of total |
| --- | --- | --- |
| North America | 42.1 | 44.2% |
| EMEA | 27.8 | 29.2% |
| APAC | 18.4 | 19.3% |
| LATAM | 6.9 | 7.2% |
"""

SUPPORT = """| Plan | Seats included | Monthly price per seat | First response time |
| --- | --- | --- | --- |
| Basic | 1 to 10 seats | $12 | 24 hours |
| Business | 11 to 100 seats | $9 | 4 hours |
| Enterprise | 101 or more seats | $7 | 30 minutes |
"""


def test_parse_tables_reads_headers_and_rows():
    (headers, rows), = parse_tables(REVENUE)
    assert headers == ["Region", "Revenue", "Share of total"]
    assert rows[0] == ["North America", "42.1", "44.2%"] and len(rows) == 4


def test_highest_value_is_confirmed():
    claim = "North America had the highest revenue in Q3, with 42.1 million, which is 44.2% of the total."
    assert check_table_claim(claim, {"E1": REVENUE}) is not None


def test_wrong_row_is_not_confirmed():
    assert check_table_claim("EMEA had the highest revenue in Q3.", {"E1": REVENUE}) is None


def test_lowest_value_is_confirmed():
    assert check_table_claim("LATAM had the lowest revenue in Q3 at 6.9 million.", {"E1": REVENUE}) is not None
    assert check_table_claim("APAC had the lowest revenue in Q3.", {"E1": REVENUE}) is None


def test_number_from_another_row_is_rejected():
    assert check_table_claim("North America had the highest revenue at 27.8 million.", {"E1": REVENUE}) is None


def test_added_unsupported_words_are_rejected():
    claim = "North America had the highest revenue because of aggressive marketing campaigns and strong partnerships."
    assert check_table_claim(claim, {"E1": REVENUE}) is None


def test_time_units_are_compared_in_minutes():
    # 30 minutes is faster than 4 hours even though 30 > 4 as raw numbers
    assert check_table_claim("The Enterprise plan has the fastest first response time of 30 minutes.", {"E1": SUPPORT}) is not None
    assert check_table_claim("The Business plan has the fastest first response time.", {"E1": SUPPORT}) is None


def test_non_superlative_claims_are_left_to_the_normal_verifier():
    assert check_table_claim("North America had revenue of 42.1 million.", {"E1": REVENUE}) is None
