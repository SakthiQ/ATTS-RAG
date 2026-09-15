"""Checks that knowledge ingestion is complete and the index is healthy.

    python scripts/check_index.py

Passes when the registry, BM25 index and Chroma hold exactly the same chunks, and every
document carries the trust metadata added by the poisoning scan. Exits with code 1 otherwise.
Read-only: nothing is changed.
"""
import os
import sys

# Allow running as a script from the repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rag.vectorstore import VectorStoreManager


def main():
    vsm = VectorStoreManager()
    report = vsm.consistency_report()

    print("Index stores:")
    for key, value in report.items():
        print(f"  {key:30} {value}")

    print("\nDocuments:")
    if not vsm.registry:
        print("  (none ingested yet)")
    old_entries = 0
    for entry in vsm.registry.values():
        has_trust = "source_tier" in entry and "version" in entry
        old_entries += not has_trust
        print(
            f"  {'OK ' if has_trust else 'OLD'} {entry['filename'][:50]:50} "
            f"tier={entry.get('source_tier', '-')} v{entry.get('version', '-')} "
            f"indexed={entry['chunk_count']} quarantined={len(entry.get('quarantined', []))}"
        )

    passed = report["consistent"] and old_entries == 0
    if passed:
        print("\nRESULT: PASS - ingestion complete, all three stores agree.")
    else:
        print("\nRESULT: FAIL - stop the backend and run:  python scripts/reingest_corpus.py --rebuild --tier official")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
