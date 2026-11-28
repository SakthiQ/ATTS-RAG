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


# ---------- After a loader or chunker change: every chunk-set hash changes ----------

def long_text(topic, changed=False):
    """Eight paragraphs, several chunks; `changed` rewrites the first paragraph, as a loader change would."""
    paragraphs = [" ".join(f"{topic} clause {i}.{j} applies to team {i} on working day {j}." for j in range(20))
                  for i in range(8)]
    if changed:
        paragraphs[0] = paragraphs[0].replace("applies to", "covers")
    return "\n\n".join(paragraphs)


@pytest.fixture()
def changed_corpus(tmp_path):
    """Documents indexed, then their text read differently: one deleted version, one kept, one direct file."""
    data, persist = tmp_path / "data", tmp_path / "chroma_db"
    policy = write(data / "policy.txt", long_text("Travel"))
    v1 = write(data / "uploads" / "tenant_a" / "f1" / "guide.txt", long_text("Expense"))
    v2 = write(data / "uploads" / "tenant_a" / "f2" / "guide.txt", long_text("Expense") + "\n\nVersion two adds receipts.")
    vsm = VectorStoreManager(persist_directory=str(persist))
    loader, chunker, g = DocumentLoader(), DocumentChunker(), guard()
    ingest_file(policy, vsm, g, loader, chunker, source_tier="official", document_id="travel_policy")
    old = ingest_file(v1, vsm, g, loader, chunker, source_tier="verified_internal", document_id="expense_guide",
                      uploaded_by="tara", tenant_id="tenant_a", acl_tier=2)
    ingest_file(v2, vsm, g, loader, chunker, source_tier="verified_internal", document_id="expense_guide",
                uploaded_by="tara", tenant_id="tenant_a", acl_tier=2)
    vsm.delete_document(old["content_hash"])   # an admin deleted version 1; its file stays on disk
    assert len(DocumentChunker().chunk_documents(loader.load_any(policy))) >= 4
    write(data / "policy.txt", long_text("Travel", changed=True))
    write(data / "uploads" / "tenant_a" / "f1" / "guide.txt", long_text("Expense", changed=True))
    write(data / "uploads" / "tenant_a" / "f2" / "guide.txt",
          long_text("Expense", changed=True) + "\n\nVersion two adds receipts.")
    write(data / "notes.txt", "Never indexed: a scratch note left in the data folder.")
    work = tmp_path / "index"
    shutil.copytree(persist, work)
    return data, work


def test_a_rebuild_after_an_ingestion_change_keeps_settings_and_deletions(changed_corpus, capsys):
    data, persist = changed_corpus
    load_script().run(["--rebuild", "--registered-only", "--data-dir", str(data), "--persist-dir", str(persist)],
                      guard=guard())
    kept = [(t, d, tier, acl, by) for t, d, _, tier, acl, by in entries(persist)]
    assert kept == [
        (DEFAULT_TENANT, "travel_policy", "official", 1, "anonymous"),
        ("tenant_a", "expense_guide", "verified_internal", 2, "tara"),
    ]
    assert consistent(persist)
    out = capsys.readouterr().out
    assert "matched to its earlier record" in out
    assert "SKIPPED uploads" in out and "f1" in out, "the deleted version stays deleted"
    assert "SKIPPED notes.txt" in out, "--registered-only leaves files the index never had out"


def test_without_registered_only_new_files_in_data_are_added(changed_corpus):
    data, persist = changed_corpus
    load_script().run(["--rebuild", "--data-dir", str(data), "--persist-dir", str(persist)], guard=guard())
    assert (DEFAULT_TENANT, "notes", 1, "unknown", 1, "admin-cli") in entries(persist)


def test_registered_only_and_include_unregistered_are_refused_together(changed_corpus):
    data, persist = changed_corpus
    with pytest.raises(SystemExit):
        load_script().run(["--registered-only", "--include-unregistered", "--data-dir", str(data),
                           "--persist-dir", str(persist)], guard=guard())
