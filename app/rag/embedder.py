import functools
from typing import List, Tuple
from langchain_huggingface import HuggingFaceEmbeddings

class DocumentEmbedder:
    """Handles converting text chunks into numerical vector embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # This model runs locally on your CPU/GPU
        # The first time you run this, it will download (~80MB)
        self.client = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}, # Change to 'cuda' if you have an NVIDIA GPU
            encode_kwargs={'normalize_embeddings': True} # Better for cosine similarity
        )

    @functools.lru_cache(maxsize=512)
    def _cached_embed_query(self, text: str) -> Tuple[float, ...]:
        """LRU cached query vector lookup."""
        return tuple(self.client.embed_query(text))

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single string (the user's question) with sub-millisecond LRU cache."""
        return list(self._cached_embed_query(text))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of strings (the document chunks)."""
        return self.client.embed_documents(texts)

# Example Usage
if __name__ == "__main__":
    embedder = DocumentEmbedder()
    # print(len(vector)) # Should be 384
