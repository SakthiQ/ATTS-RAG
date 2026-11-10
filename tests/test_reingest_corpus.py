"""scripts/reingest_corpus.py: a rebuild re-creates every document with its tenant, settings and version, and never
brings back a deleted upload."""
import importlib.util
import json
import pathlib
import shutil

import pytest

from app.rag.chunker import DocumentChunker
from app.rag.ingestion import ingest_file
from app.rag.ingestion_guard import IngestionGuard
from app.rag.loader import DocumentLoader
from app.rag.vectorstore import DEFAULT_TENANT, VectorStoreManager

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location("reingest_corpus", ROOT / "scripts" / "reingest_corpus.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def guard():
    return IngestionGuard(str(ROOT / "config" / "injection_patterns.yaml"), judge=lambda text: {"injection": False, "confidence": 0.0})


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


@pytest.fixture()
def corpus(tmp_path):
    """A data folder and an index built the way the app builds it, then one upload deleted by an admin."""
    data, persist = tmp_path / "data", tmp_path / "chroma_db"
    legacy = write(data / "handbook.txt", "The office opens at 8 am and closes at 6 pm on weekdays.")
    v1 = write(data / "uploads" / "tenant_a" / "f1" / "pay.txt", "Pay bands: engineers earn 90k to 110k a year.")
    v2 = write(data / "uploads" / "tenant_a" / "f2" / "pay.txt", "Pay bands: engineers earn 95k to 120k a year.")
    gone = write(data / "uploads" / "tenant_b" / "f3" / "plan.txt", "Merger plan: acquire Globex in the third quarter.")
    vsm = VectorStoreManager(persist_directory=str(persist))
    loader, chunker, g = DocumentLoader(), DocumentChunker(), guard()
    ingest_file(legacy, vsm, g, loader, chunker, source_tier="official", document_id="handbook")
    ingest_file(v1, vsm, g, loader, chunker, source_tier="verified_internal", document_id="pay_bands",
                uploaded_by="tara", tenant_id="tenant_a", acl_tier=2)
    ingest_file(v2, vsm, g, loader, chunker, source_tier="verified_internal", document_id="pay_bands",
                uploaded_by="tara", tenant_id="tenant_a", acl_tier=3)
    deleted = ingest_file(gone, vsm, g, loader, chunker, document_id="merger", tenant_id="tenant_b")
    vsm.delete_document(deleted["content_hash"])   # the admin deleted it; the file stays on disk
    # This process keeps the index files open (Windows then refuses to move them), so the script works on a copy,
    # as it would on an index no running server holds
    work = tmp_path / "index"
    shutil.copytree(persist, work)
    return data, work


def entries(persist):
    with open(pathlib.Path(persist) / "document_registry.json", encoding="utf-8") as f:
        registry = json.load(f)
    return sorted((e["tenant_id"], e["document_id"], e["version"], e["source_tier"], e["acl_tier"], e["uploaded_by"])
                  for e in registry.values())


def consistent(persist):
    return VectorStoreManager(persist_directory=str(persist)).consistency_report()["consistent"]


def test_a_rebuild_restores_every_document_with_its_tenant_settings_and_version(corpus, capsys):
    data, persist = corpus
    before = entries(persist)
    load_script().run(["--rebuild", "--data-dir", str(data), "--persist-dir", str(persist)], guard=guard())
    after = entries(persist)
    assert after == before == [
        (DEFAULT_TENANT, "handbook", 1, "official", 1, "anonymous"),
        ("tenant_a", "pay_bands", 1, "verified_internal", 2, "tara"),
        ("tenant_a", "pay_bands", 2, "verified_internal", 3, "tara"),
    ]
    assert consistent(persist)
    out = capsys.readouterr().out
    assert "SKIPPED" in out and "plan.txt" in out, "the deleted upload is reported, not re-indexed"
    assert list(pathlib.Path(persist).parent.glob("index.backup-*")), "the old index is kept"


def test_deleted_uploads_come_back_only_when_asked(corpus):
    data, persist = corpus
    load_script().run(["--rebuild", "--include-unregistered", "--data-dir", str(data), "--persist-dir", str(persist)],
                      guard=guard())
    assert ("tenant_b", "plan", 1, "unknown", 1, "admin-cli") in entries(persist)


def test_an_in_place_run_replaces_entries_without_duplicates(corpus):
    data, persist = corpus
    before = entries(persist)
    load_script().run(["--data-dir", str(data), "--persist-dir", str(persist)], guard=guard())
    assert entries(persist) == before and consistent(persist)
