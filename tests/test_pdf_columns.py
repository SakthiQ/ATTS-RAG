"""Multi-column pages must be read column by column, and ordinary pages must be left alone.

pdfplumber reads a page line by line across its full width, which interleaved side-by-side slide
columns ("Constrain LLM outputs to Process PHI using") and left their facts unfindable.
"""
from app.rag.loader import DocumentLoader


def _write_pdf(path, lines):
    """Writes a one-page PDF. lines: (x, y, font_size, text) in points on a 612x792 page, Helvetica."""
    ops = []
    for x, y, size, text in lines:
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        ops.append(f"BT /F1 {size} Tf {x} {y} Td ({safe}) Tj ET")
    stream = "\n".join(ops).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    path.write_bytes(bytes(out))


def _text(path):
    return "\n".join(d["content"] for d in DocumentLoader().load_pdf(str(path)))


LEFT = ["Stepwise implementation starts here", "Text preprocessing cleans the input", "Medical NER extracts the entities",
        "Retrieval searches the knowledge base", "Generation writes the explanation"]
RIGHT = ["Safety controls constrain outputs", "Answers must cite retrieved evidence", "A disclaimer ships with each answer",
         "Provenance is logged for review", "Clinicians can audit every step"]


def test_two_column_slide_is_read_column_by_column(tmp_path):
    lines = [(50, 740, 22, "Implementation Methodology")]
    for i, (left, right) in enumerate(zip(LEFT, RIGHT)):
        y = 690 - i * 16
        lines += [(50, y, 11, left), (330, y, 11, right)]
    _write_pdf(tmp_path / "slide.pdf", lines)
    text = _text(tmp_path / "slide.pdf")

    # Each column's lines stay together and in order, instead of alternating left/right
    positions = [text.index(line) for line in LEFT + RIGHT]
    assert positions == sorted(positions)
    assert "Stepwise implementation starts here Safety controls" not in text


def test_single_column_page_is_unchanged(tmp_path):
    prose = [
        "This report explains how the detector turns clinical language into plain explanations for patients.",
        "It identifies medical terms, retrieves definitions, and generates a faithful summary of each finding.",
        "The system explains documented findings only, and never diagnoses diseases or prescribes treatment.",
        "Every explanation carries citations to the passages it was generated from, for clinician review.",
    ]
    _write_pdf(tmp_path / "prose.pdf", [(50, 700 - i * 16, 11, line) for i, line in enumerate(prose)])
    text = _text(tmp_path / "prose.pdf")
    for line in prose:
        assert line in text


def test_table_of_contents_is_not_split_from_its_page_numbers(tmp_path):
    entries = ["Executive Summary", "The Implementation Gap", "Five Design Principles", "Data Quality", "Getting Started"]
    lines = [(50, 740, 22, "Table of Contents")]
    for i, entry in enumerate(entries):
        y = 700 - i * 16
        lines += [(50, y, 11, entry), (520, y, 11, str(3 + i * 4))]
    _write_pdf(tmp_path / "toc.pdf", lines)
    text = _text(tmp_path / "toc.pdf")
    assert "Executive Summary 3" in text
    assert "Getting Started 19" in text
