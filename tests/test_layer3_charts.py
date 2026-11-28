"""Chart check v2 (app/rag/layer3/generation.py, CHART_V2): a chart is released only if its points are ONE series of
the evidence. The cases in tests/data/chart_cases.json are measured by scripts/eval_charts.py; here each case must get
exactly its recorded outcome, and each rule has its own test."""
import json
import os

import pytest

from app.rag.layer3 import generation
from app.rag.layer3.contracts import ChartPoint, ChartSpec

DATA = json.load(open(os.path.join(os.path.dirname(__file__), "data", "chart_cases.json"), encoding="utf-8"))


def chart(type_, *points, title="Chart"):
    return ChartSpec(type=type_, title=title, data=[ChartPoint(label=l, value=v) for l, v in points])


def check(spec, *passages):
    return generation.verify_chart(spec, {f"E{i + 1}": p for i, p in enumerate(passages)}, "")


def _case_params():
    out = []
    for c in DATA["cases"]:
        marks = []
        if c["expect"] == "refused_until_cp3":
            marks = [pytest.mark.xfail(strict=True, reason="checkpoint 3: over-time charts from a table row and 'Phase N' "
                                                           "labels are not allowed yet")]
        out.append(pytest.param(c, id=c["id"], marks=marks))
    return out


@pytest.mark.parametrize("case", _case_params())
def test_every_chart_case_gets_its_recorded_outcome(case):
    spec = chart(case["type"], *[tuple(p) for p in case["points"]])
    out = generation.verify_chart(spec, {f"E{i + 1}": DATA["passages"][p] for i, p in enumerate(case["passages"])}, "")
    points = [(p.label, p.value) for p in spec.data]
    expect = case["expect"]
    if expect == "reject":
        assert out is None
    elif expect in ("release", "refused_until_cp3"):
        assert out is not None and out.type == case["type"] and [(p.label, p.value) for p in out.data] == points
    elif expect == "release_as_bar":
        assert out is not None and out.type == "bar" and [(p.label, p.value) for p in out.data] == points
    elif expect == "release_reordered":
        labels = [p.label for p in out.data]
        assert out.type == "line" and labels == sorted(labels) and sorted(p.value for p in out.data) == sorted(v for _, v in points)
    else:
        raise AssertionError(f"unknown expectation {expect}")


REVENUE = ("| Region | Q1 | Q3 | Growth |\n| --- | --- | --- | --- |\n| North America | 4.10M | 4.60M | 12% |\n"
           "| EMEA | 2.10M | 2.55M | 21% |\n| APAC | 3.10M | 4.20M | 35% |\n| Total | 9.30M | 11.35M | 22% |")


def test_an_unchanged_chart_is_returned_as_is():
    spec = chart("bar", ("North America", 4.60), ("EMEA", 2.55), ("APAC", 4.20))
    assert check(spec, REVENUE) is spec


def test_r1_points_from_two_passages_are_rejected():
    other = "| Region | Revenue 2023 |\n| --- | --- |\n| EMEA | 9.80M |\n| APAC | 1.10M |"
    assert check(chart("bar", ("EMEA", 9.80), ("APAC", 4.20)), REVENUE, other) is None
    assert check(chart("bar", ("EMEA", 9.80), ("APAC", 1.10)), REVENUE, other) is not None


def test_r2_values_from_two_columns_are_rejected():
    assert check(chart("bar", ("North America", 4.60), ("EMEA", 21)), REVENUE) is None        # revenue and growth
    assert check(chart("bar", ("North America", 4.10), ("EMEA", 2.55)), REVENUE) is None      # Q1 and Q3


def test_r2_a_cell_with_a_range_or_several_numbers_holds_no_value():
    plan = "| Phase | Duration |\n| --- | --- |\n| Assessment | 2-3 weeks |\n| Build | 1 week |"
    assert check(chart("bar", ("Assessment", 2), ("Build", 1)), plan) is None


def test_r3_figures_in_different_units_are_rejected():
    text = "APAC grew 35% this year, while EMEA revenue reached 2.5 million."
    assert check(chart("bar", ("APAC", 35), ("EMEA", 2.5)), text) is None


def test_r4_a_total_is_never_shown_next_to_its_parts():
    assert check(chart("bar", ("North America", 4.60), ("EMEA", 2.55), ("Total", 11.35)), REVENUE) is None
    assert check(chart("pie", ("North America", 4.60), ("EMEA", 2.55), ("APAC", 4.20), ("Total", 11.35)), REVENUE) is None


def test_r5_list_numbering_is_never_a_value():
    table = ("| Pitfall | Description | Share of failures (%) |\n| --- | --- | --- |\n| 1 | Treating pilots as the end goal | 70 |\n"
             "| 2 | Ignoring compliance early | 60 |\n| 3 | Forcing workflow changes | 55 |")
    assert check(chart("bar", ("Treating pilots as the end goal", 1), ("Forcing workflow changes", 3)), table) is None
    assert check(chart("bar", ("Treating pilots as the end goal", 70), ("Forcing workflow changes", 55)), table) is not None


def test_r6_a_pie_is_kept_only_for_one_whole_and_otherwise_shown_as_a_bar():
    shares = "| Segment | Share (%) |\n| --- | --- |\n| Enterprise | 45 |\n| Mid-market | 30 |\n| Small business | 25 |"
    assert check(chart("pie", ("Enterprise", 45), ("Mid-market", 30), ("Small business", 25)), shares).type == "pie"
    parts = check(chart("pie", ("North America", 4.60), ("EMEA", 2.55), ("APAC", 4.20)), REVENUE)
    assert parts.type == "pie"                                                    # all the parts of the Total row
    some = check(chart("pie", ("North America", 4.60), ("EMEA", 2.55)), REVENUE)
    assert some.type == "bar"                                                     # two of the three parts
    growth = check(chart("pie", ("North America", 12), ("EMEA", 21), ("APAC", 35)), REVENUE)
    assert growth.type == "bar" and [p.value for p in growth.data] == [12, 21, 35]


def test_r7_each_figure_of_a_sentence_stays_with_its_own_label():
    text = "In 2025 APAC grew 35% while EMEA grew 21%."
    assert check(chart("bar", ("APAC", 35), ("EMEA", 21)), text) is not None
    assert check(chart("bar", ("EMEA", 35), ("APAC", 21)), text) is None          # swapped
    years = "Headcount rose from 120 staff in 2024 to 150 staff in 2025."
    assert check(chart("bar", ("2024", 120), ("2025", 150)), years) is not None   # figure before its label
    assert check(chart("bar", ("2024", 150), ("2025", 120)), years) is None


def test_r7_a_label_written_twice_in_one_line_is_not_paired():
    text = "APAC grew 35%, and APAC margins reached 20% while EMEA grew 21%."
    assert check(chart("bar", ("APAC", 20), ("EMEA", 21)), text) is None


def test_r8_a_line_needs_time_labels_and_is_put_in_the_documents_order():
    trend = "| Quarter | Revenue |\n| --- | --- |\n| Q1 2026 | 81.5 |\n| Q2 2026 | 88.3 |\n| Q3 2026 | 95.2 |"
    ordered = check(chart("line", ("Q3 2026", 95.2), ("Q1 2026", 81.5), ("Q2 2026", 88.3)), trend)
    assert ordered.type == "line" and [p.label for p in ordered.data] == ["Q1 2026", "Q2 2026", "Q3 2026"]
    regions = check(chart("line", ("North America", 4.60), ("EMEA", 2.55)), REVENUE)
    assert regions.type == "bar"


def test_chart_check_v2_can_be_switched_off_to_reproduce_earlier_reports(monkeypatch):
    mixed = chart("bar", ("North America", 4.60), ("EMEA", 21))
    assert check(mixed, REVENUE) is None
    monkeypatch.setattr(generation, "CHART_V2", False)
    assert check(mixed, REVENUE) is not None
