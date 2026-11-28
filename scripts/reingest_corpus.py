"""Re-ingests the document files under data/ through the scanning pipeline.

    python scripts/reingest_corpus.py --rebuild
    python scripts/reingest_corpus.py --rebuild --tier official     (tier for files the index has no record of)

Two kinds of files are read:
  * data/uploads/<tenant>/<random id>/<file>   files uploaded through the app or API. They are re-indexed into the
                                               tenant named by their folder, with the document ID, source tier,
                                               clearance tier and uploader recorded for them in the index. An upload
                                               the index has no record of was deleted by an admin (or never finished
                                               indexing) and is skipped, so a rebuild never brings back a deleted
                                               document; --include-unregistered indexes them anyway.
  * data/<file>                                files placed directly in data/ (and uploads made before uploads had
                                               their own folders). They go to the default tenant; their recorded
                                               settings are reused when the index knows them, otherwise they get
                                               --tier, clearance 1 and a document ID derived from the file name.

A file is matched to its record by its content hash, so renamed files and older versions keep their own settings,
and versions are re-indexed oldest first so they get their old version numbers back. After a change to the loader or
chunker the content hashes all change; a file is then matched to the record of its tenant that holds most of its
chunks (at least half), preferring the same file name, so a rebuild after an ingestion change keeps every setting.

--registered-only re-indexes only the documents the index already has, and leaves out files in data/ it has no
record of (they would otherwise be added with --tier):
    python scripts/reingest_corpus.py --rebuild --registered-only

--rebuild moves the current chroma_db folder to a timestamped backup and builds a fresh index from the files,
reading the records from the backup. Use it when scripts/check_index.py reports that the stores disagree: stale
chunks without ids cannot be removed any other way. Without --rebuild, each file's existing entry is replaced in
place, which only works on a consistent index, so the script refuses to run otherwise.

Stop the API server first: this script writes the same Chroma, BM25 and registry files.
"""
import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

# Allow running as a script from the repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rag.chunker import DocumentChunker  # noqa: E402
from app.rag.ingestion import ingest_file, slugify_document_id  # noqa: E402
from app.rag.ingestion_guard import IngestionGuard  # noqa: E402
from app.rag.loader import DocumentLoader  # noqa: E402
from app.rag.trust_policy import DEFAULT_TIER, SOURCE_TIERS  # noqa: E402
from app.rag.vectorstore import DEFAULT_ACL_TIER, DEFAULT_TENANT, VectorStoreManager  # noqa: E402

PERSIST_DIR = "chroma_db"
UPLOAD_SUBDIR = "uploads"
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MIN_CHUNK_OVERLAP = 0.5   # share of a changed file's chunks its earlier record must hold to be taken as the same document


@dataclass
class SourceFile:
    path: str
    tenant_id: str
    uploaded: bool            # under data/uploads/<tenant>/...


def collect_files(data_dir: str) -> List[SourceFile]:
    """Document files directly in data_dir (default tenant) and under data_dir/uploads/<tenant>/ (that tenant)."""
    files = []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in ALLOWED_EXTENSIONS:
            files.append(SourceFile(path, DEFAULT_TENANT, uploaded=False))
    uploads = os.path.join(data_dir, UPLOAD_SUBDIR)
    if os.path.isdir(uploads):
        for tenant in sorted(os.listdir(uploads)):
            tenant_dir = os.path.join(uploads, tenant)
            if not os.path.isdir(tenant_dir):
                continue
            for root, _, names in os.walk(tenant_dir):
                for name in sorted(names):
                    if os.path.splitext(name)[1].lower() in ALLOWED_EXTENSIONS:
                        files.append(SourceFile(os.path.join(root, name), tenant, uploaded=True))
    return files


def load_registry(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def match_changed_files(unmatched: Dict[int, List[str]], files: List[SourceFile], records: Dict[str, Any],
                        used: set, vsm: VectorStoreManager) -> Dict[int, tuple]:
    """Matches files whose chunks changed (new loader or chunker code) to their earlier records.

    The registry key is a hash of all of a document's chunks, so any change to how files are read gives every
    document a new key. A code change alters only some chunks, though, and the registry keeps a hash per chunk: a
    file is matched to the unused record of its tenant that holds the largest share of its chunks (at least
    MIN_CHUNK_OVERLAP), preferring a record with the same file name. Returns {file index: (record key, share)}."""
    candidates = []
    for i, chunk_texts in unmatched.items():
        f = files[i]
        hashes = {vsm._calculate_hash(t) for t in chunk_texts}
        for key, record in records.items():
            if key in used or record.get("tenant_id", DEFAULT_TENANT) != f.tenant_id:
                continue
            share = len(hashes & set(record.get("chunk_hashes", {}).values())) / len(hashes)
            if share >= MIN_CHUNK_OVERLAP:
                same_name = record.get("filename") == os.path.basename(f.path)
                candidates.append((same_name, share, i, key))
    matches: Dict[int, tuple] = {}
    for same_name, share, i, key in sorted(candidates, reverse=True):
        if i not in matches and key not in used:
            matches[i] = (key, share)
            used.add(key)
    return matches


def plan(files: List[SourceFile], records: Dict[str, Any], vsm: VectorStoreManager, loader, chunker,
         default_tier: str, include_unregistered: bool = False, registered_only: bool = False) -> List[Dict[str, Any]]:
    """One job per file to index, with the settings to index it with, oldest version first. Files skipped
    (deleted uploads, and with registered_only every file the index has no record of) come back with "skip" set
    and the reason."""
    loaded: Dict[int, tuple] = {}
    used: set = set()
    for i, f in enumerate(files):
        chunks = chunker.chunk_documents(loader.load_any(f.path))
        content_hash = vsm.document_hash(chunks, f.tenant_id) if chunks else None
        loaded[i] = (chunks, content_hash)
        if content_hash in records:
            used.add(content_hash)
    unmatched = {i: [c["content"] for c in chunks] for i, (chunks, h) in loaded.items() if chunks and h not in records}
    changed = match_changed_files(unmatched, files, records, used, vsm)

    jobs = []
    for i, f in enumerate(files):
        chunks, content_hash = loaded[i]
        if not chunks:
            jobs.append({"file": f, "skip": "no extractable text"})
            continue
        record_key, share = (content_hash, 1.0) if content_hash in records else changed.get(i, (None, 0.0))
        record = records.get(record_key) if record_key else None
        if record is None and registered_only:
            jobs.append({"file": f, "skip": "not in the index; run without --registered-only to add it"})
            continue
        if record is None and f.uploaded and not include_unregistered:
            jobs.append({"file": f, "skip": "not in the index (deleted, or never indexed); use --include-unregistered"})
            continue
        record = record or {}
        jobs.append({
            "file": f,
            "content_hash": content_hash,
            "known": bool(record),
            "record_key": record_key if record else None,
            "matched_share": share if record and record_key != content_hash else None,
            "document_id": record.get("document_id") or slugify_document_id(f.path),
            "source_tier": record.get("source_tier", default_tier),
            "acl_tier": int(record.get("acl_tier", DEFAULT_ACL_TIER)),
            "uploaded_by": record.get("uploaded_by", "admin-cli"),
            "order": (record.get("version", 1), record.get("ingested_at", ""), os.path.getmtime(f.path)),
        })
    # Versions of a document must be re-created in their original order to keep their numbers
    jobs.sort(key=lambda j: j.get("order", (0, "", 0)))
    return jobs


def run(argv: Optional[List[str]] = None, guard: Optional[IngestionGuard] = None) -> List[Dict[str, Any]]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", default=DEFAULT_TIER, choices=list(SOURCE_TIERS),
                        help="source tier for files the index has no record of (default: %(default)s)")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--persist-dir", default=PERSIST_DIR)
    parser.add_argument("--rebuild", action="store_true", help="Back up the index and build a fresh one from the files")
    parser.add_argument("--include-unregistered", action="store_true",
                        help="also index uploads the index has no record of (normally skipped: they were deleted)")
    parser.add_argument("--registered-only", action="store_true",
                        help="only re-index documents the index already has; files in data/ it has no record of are "
                             "left out instead of being added")
    args = parser.parse_args(argv)
    if args.registered_only and args.include_unregistered:
        parser.error("--registered-only and --include-unregistered contradict each other")
    persist = args.persist_dir

    if args.rebuild and os.path.exists(persist):
        backup = f"{persist}.backup-{datetime.now():%Y%m%d-%H%M%S}"
        try:
            shutil.move(persist, backup)
        except PermissionError:
            sys.exit(f"Could not move {persist}: it is in use. Stop the API server and try again.")
        print(f"Backed up the old index to {backup}")
        records = load_registry(os.path.join(backup, "document_registry.json"))
    else:
        records = load_registry(os.path.join(persist, "document_registry.json"))

    vsm = VectorStoreManager(persist_directory=persist)
    if not args.rebuild:
        report = vsm.consistency_report()
        if not report["consistent"]:
            sys.exit(f"The index stores disagree, so entries can't be replaced in place:\n{report}\nRun again with --rebuild.")

    guard = guard or IngestionGuard()
    loader, chunker = DocumentLoader(), DocumentChunker()
    jobs = plan(collect_files(args.data_dir), records, vsm, loader, chunker, args.tier, args.include_unregistered,
                args.registered_only)

    if not args.rebuild:
        # Replace in place: remove every entry first, so re-added versions get their old numbers back. A file
        # matched by its chunks has its earlier entry under the old key.
        for job in jobs:
            if not job.get("skip"):
                for key in {job["content_hash"], job.get("record_key")}:
                    if key and key in vsm.registry:
                        vsm.delete_document(key)

    for job in jobs:
        f = job["file"]
        label = os.path.relpath(f.path, args.data_dir)
        if job.get("skip"):
            print(f"SKIPPED {label}: {job['skip']}")
            continue
        summary = ingest_file(f.path, vsm, guard, loader, chunker, source_tier=job["source_tier"],
                              document_id=job["document_id"], uploaded_by=job["uploaded_by"],
                              tenant_id=f.tenant_id, acl_tier=job["acl_tier"])
        job["summary"] = summary
        if not job["known"]:
            note = "  (no earlier record: default settings)"
        elif job.get("matched_share") is not None:
            note = f"  (text changed; matched to its earlier record by {job['matched_share']:.0%} of its chunks)"
        else:
            note = ""
        print(f"{label} -> tenant {f.tenant_id}, {job['document_id']}, tier {job['source_tier']}, "
              f"clearance {job['acl_tier']}: {summary.get('status')}{note}")

    report = vsm.consistency_report()
    print("\nIndex consistency:", "OK" if report["consistent"] else report)
    return jobs


if __name__ == "__main__":
    run()
