import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
from .loader import DocumentLoader
from .chunker import DocumentChunker
from .vectorstore import VectorStoreManager
from .ingestion_guard import IngestionGuard
from .trust_policy import SOURCE_TIERS, DEFAULT_TIER

# Sources whose chunks all go to the LLM judge at upload (deep scan). The regex patterns only cover English phrasings, so
# for sources nobody vouched for the judge reads everything; trusted tiers keep the cheap pattern-triggered scan.
# Override with the DEEP_SCAN_TIERS environment variable (comma-separated; empty = never).
DEFAULT_DEEP_SCAN_TIERS = ("unknown", "untrusted")


def deep_scan_tiers() -> Tuple[str, ...]:
    raw = os.getenv("DEEP_SCAN_TIERS")
    if raw is None:
        return DEFAULT_DEEP_SCAN_TIERS
    return tuple(t.strip() for t in raw.split(",") if t.strip())


# Lowercase letters, digits, '_' and '-', starting with a letter or digit, at most 100 characters
DOCUMENT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,99}$")


def _propagate_table_quarantine(chunks: List[Dict[str, Any]]) -> Dict[str, int]:
    """If any view of a table (summary or a row) is quarantined, quarantine the whole table.

    Without this, a table with one poisoned row could have that row rejected while its summary
    and other rows are still indexed and searchable -- a partial, inconsistent table. Paragraph
    chunks have no parent_id and are never grouped by this.
    """
    groups = defaultdict(list)
    for c in chunks:
        parent_id = c["metadata"].get("parent_id")
        if parent_id:
            groups[parent_id].append(c)

    for group in groups.values():
        if any(c["metadata"]["scan_status"] == "quarantined" for c in group):
            for c in group:
                if c["metadata"]["scan_status"] != "quarantined":
                    note = c["metadata"].get("scan_notes", "")
                    c["metadata"]["scan_status"] = "quarantined"
                    c["metadata"]["scan_notes"] = (note + "; quarantined: another part of this table was flagged").strip("; ")[:300]

    counts = {"clean": 0, "flagged": 0, "quarantined": 0}
    for c in chunks:
        counts[c["metadata"]["scan_status"]] += 1
    return counts


def slugify_document_id(filename: str) -> str:
    """Derives a document ID from a filename, e.g. 'Refund Policy.pdf' -> 'refund_policy'."""
    stem = os.path.splitext(os.path.basename(filename))[0].lower()
    slug = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return slug[:100] or "document"


def ingest_file(
    file_path: str,
    vsm: VectorStoreManager,
    guard: IngestionGuard,
    loader: DocumentLoader,
    chunker: DocumentChunker,
    source_tier: str = DEFAULT_TIER,
    document_id: Optional[str] = None,
    uploaded_by: str = "anonymous",
    deep_scan: Optional[bool] = None,
) -> Dict[str, Any]:
    """Load -> chunk -> poisoning scan -> transactional index write. Returns an ingestion summary.

    deep_scan=None decides by source tier (see deep_scan_tiers); True / False forces it."""
    if source_tier not in SOURCE_TIERS:
        raise ValueError(f"Unknown source tier: {source_tier}")
    document_id = document_id or slugify_document_id(file_path)
    filename = os.path.basename(file_path)

    docs = loader.load_any(file_path)
    chunks = chunker.chunk_documents(docs)
    if not chunks:
        logger.warning(f"No extractable text in '{filename}'. Nothing ingested.")
        return {"status": "empty", "filename": filename}

    # Skip the scan (and its LLM calls) for content that is already registered
    content_hash = vsm.document_hash(chunks)
    if content_hash in vsm.registry:
        logger.info(f"Document '{filename}' (Hash: {content_hash[:8]}) already exists. Skipping.")
        return {"status": "duplicate", "content_hash": content_hash, "filename": filename}

    # Scan before indexing so quarantined chunks never reach Chroma or BM25
    embeddings = vsm.embedder.embed_documents([c["content"] for c in chunks])
    deep = (source_tier in deep_scan_tiers()) if deep_scan is None else deep_scan
    guard.scan(chunks, embeddings, deep=deep)
    scan_counts = _propagate_table_quarantine(chunks)

    summary = vsm.add_chunks(chunks, document_id=document_id, source_tier=source_tier, uploaded_by=uploaded_by)
    summary["scan"] = scan_counts
    return summary
