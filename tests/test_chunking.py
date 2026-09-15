import os
import pytest

from app.rag.loader import DocumentLoader
from app.rag.chunker import DocumentChunker, MODEL_MAX_TOKENS

PDF = "data/Beyond the Pilot - How Regulated Industries Can Successfully Scale AI.pdf"
BODY = 10.5  # body font size used by the synthetic pages


def page(label, title, body, number):
    """A synthetic page: small running label, optional large title, body lines, page-number footer."""
    lines = [(label, 8.2)]
    if title:
        lines.append((title, 27.0))
    lines += [(text, BODY) for text in body]
    lines.append((f"Page {number}", 9.0))
    return lines


@pytest.fixture(scope="module")
def chunker():
    return DocumentChunker()


# ---------- Loader: page structure ----------

def test_running_header_and_footer_are_detected():
    pages = [page("PART 1", f"Title {n}", ["Body text."], n) for n in range(1, 6)]
    headers, footers = DocumentLoader._find_running_lines(pages)
    assert headers == {"part #"}
    assert footers == {"page #"}


def test_footer_is_dropped_header_becomes_section_and_continuation_pages_inherit_it():
    pages = [
        page("PART 1", "Data Quality", ["First page body."], 1),
        page("PART 1", None, ["Continuation body."], 2),
        page("PART 2", "Governance", ["New part body."], 3),
        page("PART 2", None, ["More."], 4),
        page("PART 2", None, ["Even more."], 5),
    ]
    structured = DocumentLoader._structure_pages(pages, BODY)
    assert structured[0] == [("PART 1 > Data Quality", ["First page body."])]
    assert structured[1] == [("PART 1 > Data Quality", ["Continuation body."])]
    assert structured[2] == [("PART 2 > Governance", ["New part body."])]

    text = " ".join(line for segments in structured for _, lines in segments for line in lines)
    assert "Page" not in text and "PART" not in text


def test_wrapped_titles_are_joined_and_subheadings_start_new_segments():
    first = [("PART 1", 8.2), ("Five Design Principles", 27.0), ("that drive success", 27.0),
             ("Intro.", BODY), ("1. Instant Value", 18.0), ("Value text.", BODY),
             ("2. Integration", 18.0), ("Integration text.", BODY)]
    pages = [first] + [[("PART 1", 8.2), ("Filler.", BODY)] for _ in range(3)]
    segments = DocumentLoader._structure_pages(pages, BODY)[0]
    assert segments == [
        ("PART 1 > Five Design Principles that drive success", ["Intro."]),
        ("PART 1 > Five Design Principles that drive success > 1. Instant Value", ["Value text."]),
        ("PART 1 > Five Design Principles that drive success > 2. Integration", ["Integration text."]),
    ]


def test_small_upper_case_first_line_is_a_label_even_if_it_appears_once():
    pages = [[("ABOUT", 8.2), ("About Us", 27.0), ("We build things.", BODY)]]
    assert DocumentLoader._structure_pages(pages, BODY)[0] == [("ABOUT > About Us", ["We build things."])]


# ---------- Chunker: size and context prefix ----------

def test_chunks_fit_the_embedding_model_and_carry_their_context(chunker):
    long_text = " ".join(f"Sentence {i} explains part of the records retention policy in detail." for i in range(400))
    docs = [{"content": long_text, "metadata": {"source": "Policy Handbook.pdf", "page": 3, "section": "PART 2 > Retention"}}]
    chunks = chunker.chunk_documents(docs)
    assert len(chunks) > 5
    for c in chunks:
        assert c["content"].startswith("Policy Handbook > PART 2 > Retention\n")
        assert chunker.count_tokens(c["content"]) <= MODEL_MAX_TOKENS
        assert chunker.count_tokens(c["content"]) <= chunker.target_tokens + 5


def test_documents_without_a_section_are_prefixed_with_the_file_name(chunker):
    chunks = chunker.chunk_documents([{"content": "Leave is 20 days.", "metadata": {"source": "test_policy.txt"}}])
    assert chunks[0]["content"] == "test_policy\nLeave is 20 days."


def test_target_above_the_model_limit_is_rejected():
    with pytest.raises(ValueError):
        DocumentChunker(target_tokens=MODEL_MAX_TOKENS + 1)


# ---------- Real PDF ----------

@pytest.mark.skipif(not os.path.exists(PDF), reason="sample PDF not present")
def test_real_pdf_sections_footers_and_chunk_sizes(chunker):
    docs = DocumentLoader().load_pdf(PDF)

    lines = [line.strip() for d in docs for line in d["content"].splitlines()]
    assert not any(line.startswith("Page ") and line[5:].isdigit() for line in lines)

    # Page 16 has no title of its own: it continues Phase 1 from page 15
    page16 = [d["metadata"]["section"] for d in docs if d["metadata"]["page"] == 16]
    assert page16 and all(s.startswith("PART 2 > Phase 1: Pre-Pilot Assessment and Strategy") for s in page16)
    assert any(s.endswith("Exit Criteria") for s in page16)

    chunks = chunker.chunk_documents(docs)
    assert max(chunker.count_tokens(c["content"]) for c in chunks) <= MODEL_MAX_TOKENS
