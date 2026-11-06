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


PHASES = """| Phase | Name | Duration |
| --- | --- | --- |
| 1 | Pre-Pilot Assessment | 2-3 weeks |
| 2 | Use Case Definition | 1-2 weeks |
| 3 | Technical Implementation | 1 week |
| 4 | Pilot Execution | 4 weeks |
| 6 | Organization-Wide Deployment | 8-12 weeks |
"""


def test_a_column_with_range_cells_is_never_compared():
    """'1-2 weeks' is not one value; treating it as 1 (unscaled) next to '1 week' (scaled) confirmed a wrong 'shortest'."""
    assert check_table_claim("Use Case Definition has the shortest duration.", {"E1": PHASES}) is None
    assert check_table_claim("Technical Implementation has the shortest duration.", {"E1": PHASES}) is None
    assert check_table_claim("Organization-Wide Deployment has the longest duration.", {"E1": PHASES}) is None


def test_range_syntax_variants_are_all_treated_as_ranges():
    from app.rag.layer3.table_check import _cell_value
    for cell in ("2-3 weeks", "2 \u2013 3 weeks", "8 to 12 weeks", "50-100 users"):
        assert _cell_value(cell) is None, cell
    assert _cell_value("1 week") == 10080.0 and _cell_value("44.2%") == 44.2 and _cell_value("500+") == 500.0
