"""Word tables, Word headings and link addresses survive loading, and table questions retrieve the right table.

Before this, a Word file was read paragraph by paragraph only: its tables were dropped without a trace, and the address
behind a clickable link (in Word or PDF) was lost, keeping only the visible text.
"""
import pytest
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from app.rag.chunker import DocumentChunker
from app.rag.engine import RAGEngine
from app.rag.loader import DocumentLoader, clean_cell, with_link_address
from app.rag.vectorstore import VectorStoreManager

HYPERLINK_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink"

REFUND_ROWS = [("Plan", "Refund window", "Restocking fee"),
               ("Basic", "14 days", "0%"),
               ("Pro", "30 days", "5%"),
               ("Enterprise", "60 days", "10%")]
SUPPORT_ROWS = [("Severity", "First response", "Resolution target"),
                ("Critical", "15 minutes", "4 hours"),
                ("High", "1 hour", "1 business day"),
                ("Low", "1 business day", "5 business days")]


def add_link(paragraph, text, url):
    rid = paragraph.part.relate_to(url, HYPERLINK_REL, is_external=True)
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), rid)
    run = OxmlElement("w:r")
    t = OxmlElement("w:t")
    t.text = text
    run.append(t)
    link.append(run)
    paragraph._p.append(link)


def add_table(doc, rows):
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            table.cell(i, j).text = value
    return table


def build_policy_docx(path):
    doc = Document()
    doc.add_heading("Customer Policies", 0)
    doc.add_heading("Refunds", 1)
    p = doc.add_paragraph("Refund requests are made through the ")
    add_link(p, "refund portal", "https://example.com/refunds")
    p.add_run(". Every plan includes a refund window, and some plans charge a restocking fee on returned hardware.")
    doc.add_paragraph("Table 1: Refund windows by plan")
    add_table(doc, REFUND_ROWS)
    doc.add_paragraph("Refunds are paid to the original payment method within ten business days of approval.")
    doc.add_heading("Support", 1)
    doc.add_paragraph("Support tickets are triaged by severity. Customers on every plan can open tickets by e-mail or "
                      "through the help centre, and the on-call engineer is paged for critical issues.")
    add_table(doc, SUPPORT_ROWS)   # no caption line above it
    box = doc.add_table(rows=1, cols=1)   # a one-cell box around a note: body text, not a data table
    box.cell(0, 0).text = "Note: response times are measured in business hours unless the ticket is critical."
    doc.add_heading("Shipping", 1)
    doc.add_paragraph("Orders ship from the regional warehouse closest to the customer. Hardware returns must be "
                      "packed in the original box, and customers pay return shipping unless the item arrived damaged.")
    doc.save(path)
    return path


@pytest.fixture(scope="module")
def policy_docs(tmp_path_factory):
    path = build_policy_docx(tmp_path_factory.mktemp("docx") / "policies.docx")
    return DocumentLoader().load_any(str(path))


# ---------- Word: tables, headings, links ----------

def test_word_tables_become_table_documents_with_every_cell(policy_docs):
    tables = [d for d in policy_docs if d["metadata"]["type"] == "table"]
    assert len(tables) == 2
    for doc, rows in zip(tables, (REFUND_ROWS, SUPPORT_ROWS)):
        assert doc["table_data"]["columns"] == list(rows[0])
        assert doc["table_data"]["rows"] == [list(r) for r in rows[1:]]
        for row in rows:
            assert "| " + " | ".join(row) + " |" in doc["content"]


def test_word_table_caption_and_section(policy_docs):
    refunds, support = [d for d in policy_docs if d["metadata"]["type"] == "table"]
    assert refunds["table_data"]["caption"] == "Table 1: Refund windows by plan"
    assert refunds["metadata"]["section"] == "Customer Policies > Refunds"
    # no "Table N" line above it: the caption names the section instead of inventing a number
    assert support["table_data"]["caption"] == "Table in Customer Policies > Support"


def test_word_headings_become_sections_and_text_is_kept(policy_docs):
    body = [d for d in policy_docs if d["metadata"]["type"] == "docx"]
    sections = [d["metadata"]["section"] for d in body]
    assert "Customer Policies > Refunds" in sections
    assert "Customer Policies > Shipping" in sections
    text = "\n".join(d["content"] for d in body)
    assert "Customer Policies" not in text   # headings travel in the section, not the body
    assert "ten business days" in text and "regional warehouse" in text
    # a one-cell box is not a data table, but its text is kept
    assert "measured in business hours" in text


def test_word_link_address_is_kept(policy_docs):
    text = "\n".join(d["content"] for d in policy_docs)
    assert "refund portal (https://example.com/refunds)." in text


def test_word_merged_and_unnamed_columns(tmp_path):
    doc = Document()
    table = doc.add_table(rows=3, cols=3)
    for i, row in enumerate([("", "Q1", "Q2"), ("North", "4.1M", "4.3M"), ("South", "2.0M", "2.2M")]):
        for j, value in enumerate(row):
            table.cell(i, j).text = value
    note = doc.add_table(rows=2, cols=2)
    merged = note.cell(0, 0).merge(note.cell(0, 1))
    merged.text = "Region total"
    note.cell(1, 0).text = "All"
    note.cell(1, 1).text = "12.6M"
    path = tmp_path / "merged.docx"
    doc.save(path)
    first, second = [d for d in DocumentLoader().load_docx(str(path)) if d["metadata"]["type"] == "table"]
    assert first["table_data"]["columns"] == ["Column 1", "Q1", "Q2"]
    assert second["table_data"]["columns"] == ["Region total", "Region total"]   # merged cell repeats
    assert second["table_data"]["rows"] == [["All", "12.6M"]]


def test_empty_word_file_gives_no_documents(tmp_path):
    path = tmp_path / "empty.docx"
    Document().save(path)
    assert DocumentLoader().load_docx(str(path)) == []


# ---------- Link addresses and cell text ----------

@pytest.mark.parametrize("text,url,expected", [
    ("refund portal", "https://example.com/refunds", "refund portal (https://example.com/refunds)"),
    ("see https://example.com/refunds", "https://example.com/refunds", "see https://example.com/refunds"),
    ("see example.com/refunds", "https://www.example.com/refunds/", "see example.com/refunds"),
    ("write to us", "mailto:help@example.com", "write to us (help@example.com)"),
    ("section 3", "#section-3", "section 3"),                       # a jump inside the document
    ("run it", "javascript:alert(1)", "run it"),                    # not a web or e-mail address
    ("tracked", "https://example.com/?" + "x" * 400, "tracked"),    # too long to be useful
])
def test_with_link_address(text, url, expected):
    assert with_link_address(text, url) == expected


def test_clean_cell_keeps_markdown_rows_intact():
    assert clean_cell("Net\nrevenue") == "Net revenue"
    assert clean_cell("A | B") == "A / B"
    assert clean_cell(None) == ""
    table = DocumentLoader._format_table([["Metric", "Value"], [clean_cell("Net\nrevenue"), "4.2M"]])
    assert "| Net revenue | 4.2M |" in table


def minimal_pdf_with_link(path, text, url):
    """A one-page PDF whose single text line is covered by a URI link annotation."""
    stream = f"BT /F1 11 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> /Annots [6 0 R] >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Annot /Subtype /Link /Rect [72 714 420 733] /Border [0 0 0] /A << /S /URI /URI ("
        + url.encode() + b") >> >>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(out)
    return path


def test_pdf_link_address_is_kept(tmp_path):
    path = minimal_pdf_with_link(tmp_path / "linked.pdf",
                                 "Download the claim form from the refund portal before you post the item.",
                                 "https://example.com/refunds")
    text = "\n".join(d["content"] for d in DocumentLoader().load_pdf(str(path)))
    assert "refund portal before you post the item. (https://example.com/refunds)" in text


# ---------- Retrieval: a question about one table value finds that table ----------

QUESTIONS = [
    ("How long is the refund window on the Enterprise plan?", "Table 1: Refund windows by plan", "60 days"),
    ("What restocking fee does the Pro plan charge?", "Table 1: Refund windows by plan", "5%"),
    ("How quickly is the first response to a critical ticket?", "Table in Customer Policies > Support", "15 minutes"),
    ("What is the resolution target for a high severity issue?", "Table in Customer Policies > Support",
     "1 business day"),
]
TOP_N = 3   # distinct elements (a table counts once) the answer must be among


@pytest.fixture(scope="module")
def indexed(policy_docs, tmp_path_factory):
    chunks = DocumentChunker().chunk_documents(policy_docs)
    for c in chunks:
        c["metadata"].update({"scan_status": "clean", "anomaly_score": 0.0})
    vsm = VectorStoreManager(persist_directory=str(tmp_path_factory.mktemp("index")))
    vsm.add_chunks(chunks, document_id="policies")
    return vsm


@pytest.mark.parametrize("question,caption,value", QUESTIONS)
def test_table_question_retrieves_the_whole_right_table(indexed, question, caption, value):
    hits = RAGEngine._dedupe_by_parent(indexed.search(question, k=16))[:TOP_N]
    tables = [h for h in hits if h["metadata"].get("element_type") == "table"]
    assert any(caption in RAGEngine._element_text(h) for h in tables), \
        f"{question!r}: top {TOP_N} were {[h['content'][:60] for h in hits]}"
    match = next(h for h in tables if caption in RAGEngine._element_text(h))
    whole = RAGEngine._element_text(match)
    assert value in whole and whole.count("\n|") >= 4   # the whole table reaches the LLM, not one row
