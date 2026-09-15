"""Tests for table-aware chunking: real-ruled-table detection, whole-element chunking,
quarantine propagation across a table, and the engine's dedupe/budget assembly.
"""
import os
import pytest

from langchain_core.runnables import RunnableLambda

from app.rag.loader import DocumentLoader
from app.rag.chunker import DocumentChunker, TABLE_WHOLE_MAX_TOKENS
from app.rag.ingestion import _propagate_table_quarantine
from app.rag.engine import RAGEngine, CONTEXT_TOKEN_BUDGET, MAX_CONTEXT_ITEMS

REAL_PDF = "data/Beyond the Pilot - How Regulated Industries Can Successfully Scale AI.pdf"
TABLE_PDF = "tests/data/table_sample.pdf"

TABLE_ROWS = [
    ["North America", "4.10M", "4.35M", "4.60M", "12%"],
    ["EMEA", "2.10M", "2.30M", "2.55M", "21%"],
    ["APAC", "3.10M", "3.40M", "4.20M", "35%"],
    ["LATAM", "1.20M", "1.25M", "1.30M", "8%"],
    ["Total", "10.50M", "11.30M", "12.65M", "20%"],
]
TABLE_COLUMNS = ["Region", "Q1", "Q2", "Q3", "Growth"]


@pytest.fixture(scope="module")
def chunker():
    return DocumentChunker()


def table_doc(columns=TABLE_COLUMNS, rows=TABLE_ROWS, caption="Table 1. Quarterly revenue."):
    md_rows = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    content = f"{caption}\n| {' | '.join(columns)} |\n{md_rows}"
    return {
        "content": content,
        "metadata": {"source": "report.pdf", "page": 5, "type": "table", "section": "PART 2 > Financials"},
        "table_data": {"columns": columns, "rows": rows, "caption": caption},
    }


# ---------- Loader: the real-ruled-lines gate ----------

def test_sample_pdf_tables_have_no_real_ruling_lines_and_are_correctly_rejected():
    """Documented finding: this PDF's boxed callouts have a background fill but zero drawn lines.
    pdfplumber's default detector still fires on them and, unguarded, corrupts the text (splits
    words mid-way). The real-rules gate must reject every one of them."""
    docs = DocumentLoader().load_pdf(REAL_PDF)
    assert sum(1 for d in docs if d["metadata"]["type"] == "table") == 0

    # And the paragraph text for that region must be clean -- no mid-word split from a rejected
    # table extraction. Un-gated extraction split "quickly" into "q" / "uickly" as separate cells;
    # check for that as a standalone fragment (word boundary), not as a substring of "quickly" itself.
    import re
    page7_text = " ".join(d["content"] for d in docs if d["metadata"]["page"] == 7)
    assert not re.search(r"\buickly\b", page7_text)
    assert "quickly" in page7_text.lower()


@pytest.mark.skipif(not os.path.exists(TABLE_PDF), reason="run scripts/generate_table_fixture.py first")
def test_real_ruled_table_is_detected_extracted_cleanly_and_excluded_from_paragraph_text():
    docs = DocumentLoader().load_pdf(TABLE_PDF)
    tables = [d for d in docs if d["metadata"]["type"] == "table"]
    assert len(tables) == 1

    t = tables[0]["table_data"]
    assert t["columns"] == TABLE_COLUMNS
    assert t["rows"] == TABLE_ROWS
    assert t["caption"].startswith("Table 1.")
    assert tables[0]["metadata"]["section"] == "Regional Performance Review"

    # The table's own cell text must not leak into the paragraph documents
    paragraph_text = " ".join(d["content"] for d in docs if d["metadata"]["type"] != "table")
    assert "4.60M" not in paragraph_text
    assert "APAC delivered the" in paragraph_text  # the surrounding prose is untouched


# ---------- Chunker: whole content + summary/row views ----------

def test_table_chunking_builds_summary_and_row_views_sharing_one_parent(chunker):
    views = chunker.chunk_documents([table_doc()])
    assert len(views) == 1 + len(TABLE_ROWS)  # 1 summary + 1 per row

    parent_ids = {v["metadata"]["parent_id"] for v in views}
    assert len(parent_ids) == 1 and next(iter(parent_ids))

    summary = next(v for v in views if v["metadata"]["view_type"] == "summary")
    assert "5 rows" in summary["content"] and "Region, Q1, Q2, Q3, Growth" in summary["content"]

    apac_row = next(v for v in views if "APAC" in v["content"])
    assert "Q3=4.20M" in apac_row["content"] and "Growth=35%" in apac_row["content"]

    for v in views:
        assert v["metadata"]["element_type"] == "table"
        assert chunker.count_tokens(v["content"]) <= chunker.max_tokens  # every view still fits the embedder


def test_row_views_carry_the_row_s_position_summary_view_is_marked_minus_one(chunker):
    views = chunker.chunk_documents([table_doc()])
    summary = next(v for v in views if v["metadata"]["view_type"] == "summary")
    assert summary["metadata"]["row_index"] == -1  # not a single row

    row_views = [v for v in views if v["metadata"]["view_type"] == "row"]
    assert [v["metadata"]["row_index"] for v in row_views] == list(range(len(TABLE_ROWS)))
    apac_view = next(v for v in row_views if "APAC" in v["content"])
    assert apac_view["metadata"]["row_index"] == TABLE_ROWS.index(["APAC", "3.10M", "3.40M", "4.20M", "35%"])


def test_whole_table_content_is_stored_and_includes_the_total_row_and_caption(chunker):
    views = chunker.chunk_documents([table_doc()])
    whole = views[0]["metadata"]["whole_content"]
    assert "Table 1. Quarterly revenue." in whole
    assert "Total" in whole and "12.65M" in whole and "20%" in whole  # the row no split-chunk baseline could see
    assert whole.startswith("report >")  # context prefix present


def test_oversized_table_degrades_to_view_only_no_whole_content(chunker):
    huge_rows = [[f"Row{i}", "a" * 40, "b" * 40, "c" * 40] for i in range(200)]
    doc = table_doc(columns=["Key", "X", "Y", "Z"], rows=huge_rows, caption="Table 9. Huge table.")
    views = chunker.chunk_documents([doc])
    assert all(v["metadata"]["table_too_large"] for v in views)
    assert all(v["metadata"]["whole_content"] == "" for v in views)
    assert chunker.count_tokens(f"a"*40) < TABLE_WHOLE_MAX_TOKENS  # sanity: rows themselves are small


def test_paragraph_chunks_carry_no_parent_grouping(chunker):
    doc = {"content": "Plain paragraph text about something unrelated to any table.",
           "metadata": {"source": "x.pdf", "page": 1, "type": "pdf", "section": "Intro"}}
    chunks = chunker.chunk_documents([doc])
    assert all(c["metadata"]["element_type"] == "paragraph" and c["metadata"]["parent_id"] == "" for c in chunks)


# ---------- Ingestion: quarantine propagates across a table's views ----------

def test_one_quarantined_row_quarantines_the_whole_table(chunker):
    views = chunker.chunk_documents([table_doc()])
    for v in views:
        v["metadata"]["scan_status"] = "clean"
    views[3]["metadata"]["scan_status"] = "quarantined"  # one row (APAC) flagged

    counts = _propagate_table_quarantine(views)
    assert counts == {"clean": 0, "flagged": 0, "quarantined": len(views)}
    assert all(v["metadata"]["scan_status"] == "quarantined" for v in views)
    assert "another part of this table was flagged" in views[0]["metadata"]["scan_notes"]


def test_quarantine_does_not_cross_between_unrelated_tables(chunker):
    a = chunker.chunk_documents([table_doc(caption="Table 1.")])
    b = chunker.chunk_documents([table_doc(caption="Table 2.")])
    for v in a + b:
        v["metadata"]["scan_status"] = "clean"
    a[0]["metadata"]["scan_status"] = "quarantined"

    _propagate_table_quarantine(a + b)
    assert all(v["metadata"]["scan_status"] == "quarantined" for v in a)
    assert all(v["metadata"]["scan_status"] == "clean" for v in b)


def test_paragraphs_are_unaffected_by_quarantine_propagation(chunker):
    paragraphs = chunker.chunk_documents([{
        "content": "Some paragraph text here.",
        "metadata": {"source": "x.pdf", "page": 1, "type": "pdf", "section": "Intro"},
    }])
    for p in paragraphs:
        p["metadata"]["scan_status"] = "clean"
    counts = _propagate_table_quarantine(paragraphs)
    assert counts["clean"] == len(paragraphs)


# ---------- Engine: dedupe by parent, budgeted assembly ----------

def make_hit(content, score, element_type="table", parent_id="p1", chunk_uid=None, whole_content=""):
    return {
        "content": content,
        "rerank_score": score,
        "metadata": {
            "element_type": element_type, "parent_id": parent_id, "whole_content": whole_content,
            "chunk_uid": chunk_uid or content, "source": "x.pdf", "page": 1,
        },
    }


def test_dedupe_collapses_multiple_table_hits_to_one_keeping_the_best_ranked():
    hits = [
        make_hit("row A", 5.0, parent_id="tbl1", whole_content="WHOLE TABLE 1"),
        make_hit("row B", 3.0, parent_id="tbl1", whole_content="WHOLE TABLE 1"),
        make_hit("other para", 4.0, element_type="paragraph", parent_id="", chunk_uid="para1"),
    ]
    deduped = RAGEngine._dedupe_by_parent(hits)
    assert [d["content"] for d in deduped] == ["row A", "other para"]  # rank order kept, tbl1 collapsed


def test_dedupe_never_merges_distinct_paragraphs_even_with_empty_parent_id():
    hits = [
        make_hit("para one", 5.0, element_type="paragraph", parent_id="", chunk_uid="u1"),
        make_hit("para two", 4.0, element_type="paragraph", parent_id="", chunk_uid="u2"),
    ]
    deduped = RAGEngine._dedupe_by_parent(hits)
    assert len(deduped) == 2


def test_element_text_prefers_whole_content_for_tables():
    hit = make_hit("Row -> Region=APAC", 5.0, whole_content="FULL TABLE INCLUDING TOTAL ROW")
    assert RAGEngine._element_text(hit) == "FULL TABLE INCLUDING TOTAL ROW"


def test_element_text_falls_back_to_chunk_content_when_no_whole_content():
    hit = make_hit("Row -> Region=APAC", 5.0, whole_content="")
    assert RAGEngine._element_text(hit) == "Row -> Region=APAC"


def test_assemble_within_budget_always_includes_at_least_the_top_hit(chunker):
    engine = RAGEngine.__new__(RAGEngine)  # skip __init__ (no Ollama/Chroma needed for this pure logic)
    engine.tokenizer = chunker.tokenizer
    huge = make_hit("x", 5.0, whole_content="word " * 5000)  # far over budget alone
    used = engine._assemble_within_budget([huge])
    assert used == [huge]


def test_assemble_within_budget_stops_before_exceeding_it(chunker):
    engine = RAGEngine.__new__(RAGEngine)
    engine.tokenizer = chunker.tokenizer
    tokens_each = chunker.count_tokens("word " * 200)
    n_hits = (CONTEXT_TOKEN_BUDGET // tokens_each) + 5  # guaranteed to overflow the budget
    hits = [make_hit(f"small{i}", 5.0, whole_content="word " * 200) for i in range(n_hits)]
    used = engine._assemble_within_budget(hits)
    assert 1 <= len(used) < n_hits
    assert len(used) <= MAX_CONTEXT_ITEMS


def test_generate_final_answer_logs_degraded_table_context(chunker, monkeypatch):
    """A table that exceeded TABLE_WHOLE_MAX_TOKENS at ingestion has no whole_content. If one of
    its views is still selected for the answer, _generate_final_answer must log that loudly,
    rather than let it pass as if the model saw the complete table."""
    engine = RAGEngine.__new__(RAGEngine)
    engine.tokenizer = chunker.tokenizer
    engine.threshold = -10.0
    engine.prompt_template = "{context}\n\n{question}"
    engine.llm = RunnableLambda(lambda x: "stub answer")  # chain is built before _invoke runs
    engine.reasoning_log = []
    monkeypatch.setattr(engine, "_invoke", lambda chain, inputs: "stub answer")

    degraded = make_hit("Row -> Region=APAC", 5.0, whole_content="")
    degraded["metadata"].update({"table_rows": 300, "view_type": "row", "page": 42})

    result = engine._generate_final_answer("irrelevant question", [degraded])

    assert result["answer"] == "stub answer"  # confirms the real method ran end-to-end, not a stub
    assert any(
        "Degraded table context" in line and "page 42" in line and "300 rows" in line
        for line in result["reasoning_log"]
    )


def test_generate_final_answer_does_not_log_degradation_for_a_whole_table(chunker, monkeypatch):
    engine = RAGEngine.__new__(RAGEngine)
    engine.tokenizer = chunker.tokenizer
    engine.threshold = -10.0
    engine.prompt_template = "{context}\n\n{question}"
    engine.llm = RunnableLambda(lambda x: "stub answer")
    engine.reasoning_log = []
    monkeypatch.setattr(engine, "_invoke", lambda chain, inputs: "stub answer")

    fine = make_hit("Row -> Region=APAC", 5.0, whole_content="FULL TABLE")
    result = engine._generate_final_answer("irrelevant question", [fine])

    assert not any("Degraded table context" in line for line in result["reasoning_log"])
