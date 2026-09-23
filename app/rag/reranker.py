import functools
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

class DocumentReranker:
    """Uses a Cross-Encoder to re-rank chunks for maximum relevance."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # This model is specifically trained for ranking pairs of (Question, Text)
        # It is about 150MB and runs locally.
        self.model = CrossEncoder(model_name, max_length=512)

    @functools.lru_cache(maxsize=1024)
    def _cached_predict_pair(self, query: str, content: str) -> float:
        """Cached cross-encoder similarity score for (query, content) pair."""
        scores = self.model.predict([(query, content)])
        return float(scores[0])

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_n: int = 4) -> List[Dict[str, Any]]:
        """Scores each chunk against the query using LRU cache and returns the best ones."""
        if not chunks:
            return []

        # Predict with LRU caching for individual query-content pairs
        uncached_pairs = []
        uncached_indices = []

        for i, chunk in enumerate(chunks):
            content = chunk["content"]
            # Check if pair score is in LRU cache
            try:
                score = self._cached_predict_pair(query, content)
                chunk["rerank_score"] = score
            except Exception:
                uncached_pairs.append((query, content))
                uncached_indices.append(i)

        if uncached_pairs:
            raw_scores = self.model.predict(uncached_pairs)
            for idx, score in zip(uncached_indices, raw_scores):
                chunks[idx]["rerank_score"] = float(score)

        # Sort by score descending (highest first)
        reranked_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        return reranked_chunks[:top_n]

# Example Usage
if __name__ == "__main__":
    print("Reranker is ready.")
