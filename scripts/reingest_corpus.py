"""Re-ingests every file in data/ through the scanning pipeline.

    python scripts/reingest_corpus.py --rebuild --tier official

--rebuild moves the current chroma_db folder to a timestamped backup and builds a fresh index.
Use it when scripts/check_index.py reports that the stores disagree: stale chunks without ids
cannot be removed any other way.

Without --rebuild, each file's existing registry entry is replaced in place. That only works
on a consistent index, so the script refuses to run otherwise.

Stop the API server first: this script writes the same Chroma, BM25 and registry files.
"""
import os
import sys
import shutil
import argparse
from datetime import datetime

# Allow running as a script from the repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rag.loader import DocumentLoader
from app.rag.chunker import DocumentChunker
from app.rag.vectorstore import VectorStoreManager
from app.rag.ingestion_guard import IngestionGuard
from app.rag.ingestion import ingest_file
from app.rag.trust_policy import SOURCE_TIERS, DEFAULT_TIER

PERSIST_DIR = "chroma_db"
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", default=DEFAULT_TIER, choices=list(SOURCE_TIERS), help="Source tier to assign to every file")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--rebuild", action="store_true", help="Back up chroma_db and build a fresh index from the data folder")
    args = parser.parse_args()

    if args.rebuild and os.path.exists(PERSIST_DIR):
        backup = f"{PERSIST_DIR}.backup-{datetime.now():%Y%m%d-%H%M%S}"
        try:
            shutil.move(PERSIST_DIR, backup)
        except PermissionError:
            sys.exit(f"Could not move {PERSIST_DIR}: it is in use. Stop the API server and try again.")
        print(f"Backed up the old index to {backup}")

    vsm = VectorStoreManager(persist_directory=PERSIST_DIR)
    if not args.rebuild:
        report = vsm.consistency_report()
        if not report["consistent"]:
            sys.exit(f"The index stores disagree, so entries can't be replaced in place:\n{report}\nRun again with --rebuild.")

    guard = IngestionGuard()
    loader, chunker = DocumentLoader(), DocumentChunker()

    files = sorted(f for f in os.listdir(args.data_dir) if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS)
    for filename in files:
        if not args.rebuild:
            # Remove the existing entry for this file so it is re-registered with trust metadata
            for content_hash, entry in list(vsm.registry.items()):
                if entry.get("filename") == filename:
                    vsm.delete_document(content_hash)

        summary = ingest_file(
            os.path.join(args.data_dir, filename), vsm, guard, loader, chunker,
            source_tier=args.tier, uploaded_by="admin-cli",
        )
        print(f"{filename}: {summary}")

    report = vsm.consistency_report()
    print("\nIndex consistency:", "OK" if report["consistent"] else report)


if __name__ == "__main__":
    main()
