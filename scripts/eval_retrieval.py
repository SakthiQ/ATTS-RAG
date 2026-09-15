"""Measures retrieval quality on a question set with known answer pages.

    python scripts/eval_retrieval.py --persist-dir chroma_db --label current

Reports Recall@1, Recall@5 and MRR@10 for three retrieval modes: dense only, hybrid (vector +
BM25 fused by RRF), and hybrid followed by the cross-encoder. No LLM is involved, so results
are deterministic. A retrieved chunk counts as correct when it comes from the expected file and,
if pages are listed, from one of those pages. Read-only: the index is not modified.
"""
import os
import sys
import json
import argparse

# Allow running as a script from the repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rag.vectorstore import VectorStoreManager
from app.rag.reranker import DocumentReranker

QUESTIONS_PATH = "tests/data/retrieval_questions.json"
RERANK_CANDIDATES = 20
TOP_K = 10


def is_relevant(chunk, gold):
    md = chunk["metadata"]
    if md.get("source") != gold["source"]:
        return False
    return not gold["pages"] or md.get("page") in gold["pages"]


def first_hit_rank(results, gold):
    for rank, chunk in enumerate(results, 1):
        if is_relevant(chunk, gold):
            return rank
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--persist-dir", default="chroma_db")
    parser.add_argument("--label", default="current")
    args = parser.parse_args()

    vsm = VectorStoreManager(persist_directory=args.persist_dir)
    reranker = DocumentReranker()
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    modes = {
        "dense": lambda q: vsm.search_vector_only(q, k=TOP_K),
        "hybrid": lambda q: vsm.hybrid_search(q, k=TOP_K),
        "hybrid+rerank": lambda q: reranker.rerank(q, vsm.hybrid_search(q, k=RERANK_CANDIDATES), top_n=TOP_K),
    }

    report = {"label": args.label, "persist_dir": args.persist_dir, "questions": len(questions), "modes": {}}
    for mode, retrieve in modes.items():
        ranks = [first_hit_rank(retrieve(g["question"]), g) for g in questions]
        n = len(ranks)
        report["modes"][mode] = {
            "recall@1": round(sum(r == 1 for r in ranks) / n, 3),
            "recall@5": round(sum(r is not None and r <= 5 for r in ranks) / n, 3),
            "mrr@10": round(sum(1 / r for r in ranks if r) / n, 3),
            "misses@5": [g["question"] for g, r in zip(questions, ranks) if r is None or r > 5],
        }

    print(f"\nRetrieval evaluation: {args.label} ({len(questions)} questions, index '{args.persist_dir}')")
    print(f"| Mode | Recall@1 | Recall@5 | MRR@10 |\n| :--- | ---: | ---: | ---: |")
    for mode, m in report["modes"].items():
        print(f"| {mode} | {m['recall@1']:.3f} | {m['recall@5']:.3f} | {m['mrr@10']:.3f} |")
    for mode, m in report["modes"].items():
        if m["misses@5"]:
            print(f"\n{mode}: not in top 5 ({len(m['misses@5'])})")
            for q in m["misses@5"]:
                print(f"  - {q}")

    os.makedirs("logs", exist_ok=True)
    out = f"logs/retrieval_eval_{args.label}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
