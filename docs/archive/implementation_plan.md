# Implementation Plan - Phase 2: Advanced Retrieval

Implementing Hybrid Search and Re-ranking to improve retrieval accuracy and move towards a production-grade system.

## User Review Required
> [!IMPORTANT]
> **Performance Impact**: Hybrid search and Re-ranking add extra computation. 
> - Hybrid search requires a BM25 index on disk.
> - Re-ranking uses a Cross-Encoder model (approx. 100MB download) and adds ~0.5s of latency per query locally.

## Proposed Changes

### [Component] Retrieval Engine
We will transform the current vector-only retrieval into a robust Hybrid system.

#### [MODIFY] [vectorstore.py](file:///c:/Users/Administrator/.gemini/antigravity/playground/metallic-rocket/ask-my-docs/app/rag/vectorstore.py)
- Integrate `rank_bm25` for keyword indexing.
- Implement `hybrid_search` method using Reciprocal Rank Fusion (RRF).

#### [NEW] [reranker.py](file:///c:/Users/Administrator/.gemini/antigravity/playground/metallic-rocket/ask-my-docs/app/rag/reranker.py)
- Use `sentence-transformers` Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to score chunk relevance against the question.

### [Component] Prompting Layer

#### [MODIFY] [responder.py](file:///c:/Users/Administrator/.gemini/antigravity/playground/metallic-rocket/ask-my-docs/app/rag/responder.py)
- Integrate the Re-ranker before calling the LLM.
- Switch from hardcoded strings to YAML-loaded prompts.

## Verification Plan

### Automated Tests
- Run `test_rag.py` with a query that specifically targets keywords (e.g., "Policy 2026") to verify Hybrid Search finds it better than Vector Search alone.
- Compare Re-ranked results vs. unsorted results in logs.
