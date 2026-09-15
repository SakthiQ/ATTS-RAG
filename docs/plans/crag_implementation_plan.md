# Implementation Plan: Self-Learning Intelligent Corrective RAG (CRAG)

Build an **Intelligent Self-Learning Corrective RAG (CRAG)** system that analyzes query intent, dynamically selects from **4 RAG paradigms** (*Naive, Advanced, Modular, Graph RAG*), evaluates retrieval accuracy, and continuously improves itself from mistakes via a persistent feedback memory loop.

---

## User Review Required

> [!IMPORTANT]
> **Dependency Addition**: Graph RAG requires building a lightweight entity-relationship network during document ingestion. We will use `networkx` (pure Python) for managing in-memory and disk-persisted knowledge graphs without needing an external heavy graph database (like Neo4j).
> 
> **Self-Learning Mechanism**: The system will learn by recording query trajectories, confidence scores, evaluation metrics, and explicit user feedback (`/feedback` endpoint). It uses past low-confidence failures as negative examples to dynamically refine strategy selection thresholds.

---

## Open Questions

> [!NOTE]
> 1. Should Graph RAG entity extraction be performed automatically for all newly uploaded documents during ingestion, or triggered on-demand? *(Default plan: Automatic lightweight entity extraction during background ingestion).*
> 2. Would you like a Streamlit visual dashboard showing the RAG strategy selection (Naive/Advanced/Modular/Graph), CRAG correction steps, and self-learning feedback analytics?

---

## Proposed Changes

### Core Engine & Intelligence Layer

#### [NEW] [intent_analyzer.py](file:///c:/Users/Administrator/.gemini/antigravity/playground/metallic-rocket/Ask%20The%20Docs/app/rag/intent_analyzer.py)
- Analyzes incoming questions against document metadata.
- Classifies query intent:
  1. **Direct Extractive**: Fact-seeking, keyword-specific, single-lookup ("What is the leave policy limit?").
  2. **Deep Analytical**: Conceptual, comparison, multi-document synthesis ("Analyze how policy X impacts remote work culture").
- Determines the optimal **RAG Strategy**:
  - `NAIVE`: Basic single-pass vector retrieval for direct simple questions.
  - `ADVANCED`: Query cleaning + Hybrid Search (Dense Chroma + BM25) + Cross-Encoder Reranking for precise direct lookups.
  - `MODULAR`: Question decomposition + multi-step sub-queries for complex analytical synthesis.
  - `GRAPH`: Entity-relationship graph traversal for multi-hop connected queries.

#### [NEW] [graph_rag.py](file:///c:/Users/Administrator/.gemini/antigravity/playground/metallic-rocket/Ask%20The%20Docs/app/rag/graph_rag.py)
- Implements `GraphRAGEngine` using `networkx`.
- Extracts entities (Subjects, Objects, Concepts) and relationships from text chunks using LLM during document indexing.
- Performs sub-graph retrieval and multi-hop entity traversal when `GRAPH` strategy is selected.

#### [NEW] [learning_store.py](file:///c:/Users/Administrator/.gemini/antigravity/playground/metallic-rocket/Ask%20The%20Docs/app/rag/learning_store.py)
- Implements `LearningStore` to record execution trajectories:
  - Query string, classified intent, chosen strategy, retrieved chunk IDs, confidence score, CRAG critique outcome, user feedback (+1 / -1).
- Provides dynamic few-shot learning context to the `IntentAnalyzer` and adjusts routing threshold parameters based on historical failure modes.

#### [MODIFY] [engine.py](file:///c:/Users/Administrator/.gemini/antigravity/playground/metallic-rocket/Ask%20The%20Docs/app/rag/engine.py)
- Refactors `RAGEngine` to integrate the **Corrective RAG (CRAG)** loop:
  1. **Intent Analysis & Strategy Selection**: Determines `DIRECT` vs `ANALYTICAL` and picks `NAIVE`, `ADVANCED`, `MODULAR`, or `GRAPH`.
  2. **Strategy Execution**: Routes to corresponding retrieval pipeline.
  3. **Corrective Evaluation**:
     - Evaluates confidence score ($S$).
     - If $S \ge \text{high}$: Proceed to synthesis (**CORRECT**).
     - If $\text{low} < S < \text{high}$: Perform query refinement / sub-search (**AMBIGUOUS**).
     - If $S \le \text{low}$: Fallback to web search or Graph RAG fallback (**INCORRECT**).
  4. **Self-Correction & Learning**: Logs trajectory and learns from mistakes.

---

### Backend API & Data Pipeline

#### [MODIFY] [vectorstore.py](file:///c:/Users/Administrator/.gemini/antigravity/playground/metallic-rocket/Ask%20The%20Docs/app/rag/vectorstore.py)
- Integrates `GraphRAGEngine` trigger during chunk ingestion (`add_chunks`).
- Fixes RRF deduplication key bug so chunk metadata (source, page, id) is preserved.

#### [MODIFY] [routes.py](file:///c:/Users/Administrator/.gemini/antigravity/playground/metallic-rocket/Ask%20The%20Docs/app/routes.py)
- Updates `QueryResponse` schema to return `strategy_used`, `intent_type`, `crag_correction_action`, and `reasoning_log`.
- Adds `POST /feedback` endpoint to receive user thumbs-up/down feedback and update the `LearningStore`.
- Adds `GET /learning/stats` endpoint to monitor self-improvement analytics.

---

## Verification Plan

### Automated Tests
1. **Unit & Pipeline Tests**:
   - `python -m pytest tests/` to verify ingestion, intent classification, and vector store operations.
2. **Strategy Routing Test**:
   - Create `tests/test_crag_strategies.py` to test query routing for all 4 strategies (Naive, Advanced, Modular, Graph).
3. **Self-Learning Loop Test**:
   - Create `tests/test_learning_loop.py` to simulate submitting queries, failing evaluations, logging feedback, and verifying that routing adapts.

### Manual Verification
1. Submit direct questions (e.g. "What is the policy for leave?") and verify system selects `ADVANCED` / `NAIVE` strategy.
2. Submit complex multi-entity analytical questions and verify system selects `MODULAR` / `GRAPH` strategy.
3. Submit feedback via `/feedback` endpoint and verify learned historical adjustments in `/learning/stats`.
---
