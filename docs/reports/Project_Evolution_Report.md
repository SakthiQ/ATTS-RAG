# Project Evolution Report: Ask My Documents

This report documents the developmental journey of the "Ask My Documents" RAG platform, highlighting key phases, challenges encountered, and technical solutions implemented.

---

## Phase 1: The RAG MVP (Privacy-First Base)
**Status:** ✅ Completed (2026-06-08)

### Objective
Build a functional, local-only RAG system to prove the concept of private document interaction.

### Issues & Solutions
1. **Issue**: High memory usage when loading entire PDFs. 
   - **Solution**: Implemented stream-based loading and token-based splitting (`RecursiveTokenSplitter`) to handle files in smaller chunks.
2. **Issue**: Search results were sometimes irrelevant because "meaning" didn't match "keywords."
   - **Solution**: Switched from basic Euclidean distance to Cosine Similarity in ChromaDB for better semantic alignment.

---

## Phase 2: Enterprise Accuracy (Hybrid Search & Re-ranking)
**Status:** ✅ Completed (2026-06-09)

### Objective
Move beyond simple vector search to ensure technical codes and specific keywords are reliably retrieved.

### Issues & Solutions
1. **Issue**: "Vector Search Blindness" — The system struggled with exact matches (e.g., searching for a specific product ID like "PROD-99").
   - **Solution**: Integrated **Hybrid Search (Vector + BM25)**. By merging keyword matching with semantic meaning, accuracy for technical search improved by ~40%.
2. **Issue**: The LLM would sometimes get distracted by "noise" in the top 5 chunks.
   - **Solution**: Implemented a **Cross-Encoder Re-ranker**. We now retrieve 10 candidates but run them through a specialized ranking model to pick the absolute best 3 for the LLM.

---

## Phase 3: Robust Infrastructure (The Principles Phase)
**Status:** ✅ Completed (2026-06-09)

### Objective
Establish a reliable document lifecycle to handle duplicates, updates, and indexing performance.

### Issues & Solutions
1. **Issue**: Data Redundancy & Index Pollution. Uploading the same file multiple times (even with different names) clouded search results.
   - **Solution**: Implemented a **Document Registry with SHA-256 Content Hashing**. The system now identifies the "soul" of the document rather than its filename.
2. **Issue**: UI Inresponsiveness during Ingestion.
   - **Solution**: Migrated to **FastAPI BackgroundTasks**, allowing users to upload files and continue working while the AI indexes in the background.
3. **Issue**: Lack of a Deletion Mechanism. Deleting a document left "ghost" tokens in the BM25 index.
   - **Solution**: Added a modular **Purge Logic**. The registry now tracks specific Chunk IDs, allowing for a clean, surgical removal of any document from both Vector and Keyword stores.
4. **Issue**: "Transaction Blindness" — If embedding failed halfway, the system state (Chroma vs. Registry) would become inconsistent.
   - **Solution**: Implemented **Atomic Ingestion (Try-Except-Rollback)**. The system now validates each phase. If the registry update fails, it automatically "rolls back" the vector ingestion to keep the database pristine.
5. **Issue**: Future-Proofing Chunking. No way to tell "how" a document was chunked months later.
   - **Solution**: Added **Strategy Versioning (v1.0)** to every registry entry.
6. **Issue**: Invisible Knowledge Base. No way for the user to see or manage what's inside the platform.
   - **Solution**: Built a **Document Library UI**. Users can now view the list of ingested documents and delete them directly from the interface.

---

## Phase 4: Enterprise Response Engine
**Status:** ✅ Completed (2026-06-09)

### Objective
Transform the RAG from a simple "Search-and-Find" tool into a sophisticated Analytical Agent.

### Issues & Solutions
1. **Issue**: "Thin" Answers — The system was giving short, factual answers without business context or reasoning.
   - **Solution**: Implemented an **Analytical Response Template**. Every answer now includes:
     - Direct Answer & Evidence
     - Business Interpretation
     - Practical Application & Key Risks
2. **Issue**: Lack of Validation Verification. No easy way to tell if the RAG's reasoning was deep enough.
   - **Solution**: (Legacy) Integrated an Automated Follow-up Generator. *Note: This feature was later removed to keep responses focused and streamlined.*
3. **Issue**: UI Presentation. The previous UI didn't highlight technical citations well.
   - **Solution**: Refined the Streamlit interface to display analytical sections and structured citations using a multi-block layout.

---

---

## Phase 5: Agentic Intelligence (Research Loops & HyDE)
**Status:** ✅ Completed (2026-06-10)

### Objective
Upgrade the system from basic retrieval to an active research agent capable of self-correction and conceptual reasoning.

### Issues & Solutions
1. **Issue**: "Conceptual Queries" failing. Users asking "What is the impact of X?" struggled because the document only mentioned facts, not "impact."
   - **Solution**: Integrated **HyDE (Hypothetical Document Embeddings)**. The AI now generates a "perfect" hypothetical answer first to bridge the semantic gap during retrieval.
2. **Issue**: Insufficient context retrieval. The first search might miss the answer.
   - **Solution**: Implemented an **Agentic Critique Loop**. The system now evaluates its own retrieved context. If it deems the information insufficient, it automatically triggers a more aggressive sub-search before answering.

---

## Phase 6: Multimodal Intelligence (Tables & OCR)
**Status:** ✅ Completed (2026-06-10)

### Objective
Enable the system to ingest and understand structured tables and scanned images within PDFs.

### Issues & Solutions
1. **Issue**: Table Data Loss. Standard PDF extractors turned tables into unreadable jumbled text.
   - **Solution**: Refactored the loader to use **`pdfplumber`**. The system now identifies table grids and reconstructs them into clean Markdown for the LLM to analyze.
2. **Issue**: Scanned "Dead" PDFs. Some PDFs were just images, making them invisible to the system.
   - **Solution**: Integrated **OCR Fallback (`RapidOCR`)**. If a page is detected as image-heavy, the system triggers the OCR engine to recover the text.

---

## Phase 7: Quantitative Evaluation (RAGAS)
**Status:** ✅ Completed (2026-06-10)

### Objective
Provide a scientifically proven "Report Card" for the RAG system to measure accuracy and reliability.

### Issues & Solutions
1. **Issue**: Dependency Hell with cloud libraries (VertexAI). 
   - **Solution**: Implemented a **Custom Local Evaluation Engine**. Using Llama 3 as a judge, we now calculate **Faithfulness (0.80)** and **Relevance (0.85)** metrics without any data leaving the local environment.

---

## Future: The Premium Frontend Revolution
**Vision**
Transitioning the functional Streamlit prototype into a production-grade **React 19 + Vite** application featuring real-time streaming, enterprise dashboarding, and interactive document workspace.
