<div align="center">
  <img src="assets/logo.png" alt="Ask My Documents Logo" width="200"/>
  <h1>Ask My Documents</h1>
  <p><strong>A Privacy-First RAG Platform, evolving into ATTS-RAG — an Adaptive Threat-intelligence Trusted &amp; Secure RAG framework</strong></p>

  [![GitHub License](https://img.shields.io/github/license/SakthiQ/ask-my-docs?style=flat-square&color=blue)](https://github.com/SakthiQ/ask-my-docs/blob/main/LICENSE)
  [![Python](https://img.shields.io/badge/python-3.11+-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
  [![Ollama](https://img.shields.io/badge/Ollama-Llama3-orange?style=flat-square)](https://ollama.com/)
  [![Status](https://img.shields.io/badge/status-beta-orange?style=flat-square)]()
  [![TRL](https://img.shields.io/badge/TRL-2%20%E2%80%94%20concept%20formulated-lightgrey?style=flat-square)]()
</div>

---

## 📌 Table of Contents

- [What is Ask My Documents?](#-what-is-ask-my-documents)
- [Cutting-Edge Features](#-cutting-edge-features)
- [Architecture](#-architecture)
- [🛡️ ATTS-RAG Security Architecture](#️-atts-rag-security-architecture)
- [Quick Start](#-quick-start-5-minutes)
- [Interactive Demo](#-interactive-demo)
- [Try It Locally](#try-it-locally)
- [API Guide](#-api-guide)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [How to Update This Project on GitHub](#-how-to-update-this-project-on-github)
- [Roadmap](#-roadmap)

---

## 🎯 What is Ask My Documents?

**Ask My Documents** is an enterprise-grade, privacy-first Retrieval-Augmented Generation (RAG) system. It transforms your local PDFs, DOCX, and Markdown files into an interactive knowledge base—completely offline.

> [!IMPORTANT]
> **100% Local Logic**: No data ever leaves your machine. We use Ollama for LLM inference and Sentence-Transformers for local embeddings.

This repository is also the reference implementation for **ATTS-RAG** (Adaptive Threat-intelligence Trusted & Secure RAG), a research framework that wraps the retrieve-and-generate loop in three security layers — query-time threat screening, trust-weighted retrieval, and hallucination-aware verification — plus an adaptive control loop that tightens thresholds as a session accumulates risk. The retrieval and ingestion engine below is fully built; the three ATTS-RAG layers are in active development. See [ATTS-RAG Security Architecture](#️-atts-rag-security-architecture) for exactly what's implemented today versus planned, and [`implementation_plan.md`](implementation_plan.md) for the build-out plan.

---

## ✨ Cutting-Edge Features

| Feature | Description | Status |
| :--- | :--- | :---: |
| 🔁 **Query Expansion** | Searches with the original question plus two LLM-generated rewrites to improve recall. | ✅ |
| 🛑 **Safe Refusal** | Declines to answer when no retrieved passage clears the relevance threshold, instead of guessing. | ✅ |
| 🔍 **Hybrid Search** | Combines Semantic Vector (Chroma) + Keyword (BM25) search with Reciprocal Rank Fusion. | ✅ |
| 🧠 **Cross-Encoder** | State-of-the-art re-ranking for maximum citation accuracy. | ✅ |
| 📑 **Exact Citations** | Precise page, paragraph, and source file tracking. | ✅ |
| ⚡ **Fast Path** | Optimized retrieval for simple factual questions. | ✅ |
| 🖼️ **Multimodal** | Extraction of tables and OCR for image-heavy PDFs. | ✅ |
| 🕵️ **Ingestion Poisoning Scan** | Every upload is screened for hidden instructions (threat-feed regex + LLM judge + embedding-outlier check) before indexing. | ✅ |
| 🚦 **Layer 1 — Query Threat Gate** | Screens incoming queries for prompt injection / jailbreak attempts, with session-level risk accumulation and dynamic risk scoring. | ✅ |
| ⚖️ **Layer 2 — Trust-Weighted Ranking** | Discrete decision tree using source tier, anomaly score, content-hash provenance, and tenant authorization to tier and filter retrieved evidence. | ✅ |
| ✅ **Layer 3 — Evidence-to-Answer Verification** | Contract-constrained generation, fast-fail safety scanning, direct NLI claim verification, evidence-gap-aware failure routing, and verified answer reconstruction. | ✅ |
| 🔄 **Offline Improvement Flywheel** | Telemetry → Failure/Canary Review → Root-Cause Analysis → Dataset Curation → Model/Retrieval/Prompt Improvements → Regression Benchmark → Versioned Deployment. | ✅ Architecture Defined |

---

## 🏗️ Architecture

The complete ATTS-RAG pipeline: query processing, hybrid retrieval, ingestion poisoning scan, and three security layers wrapping the retrieve-and-generate loop.

```mermaid
flowchart TD
    subgraph UserInterface["🌐 User Interface"]
        Q["User Question"]
        U["Upload Document"]
    end

    subgraph QueryLayer["🧠 Query Processing"]
        FP{"Short query?"}
        MQ["Query Expansion<br/>(original + 2 rewrites)"]
        TH{"Best score ≥ threshold?"}
        LLM["LLM Synthesis<br/>(top 5 passages)"]
        REF["Safe Refusal"]
    end

    subgraph RetrievalEngine["🔍 Retrieval Engine"]
        VS[("Hybrid Store<br/>Chroma + BM25 · RRF")]
        RK["Cross-Encoder Rerank"]
    end

    subgraph IngestionPipeline["📥 Ingestion (incl. poisoning scan)"]
        Loader --> Chunker --> Guard["Poisoning Scan<br/>(threat feed + LLM judge + embedding outlier)"] --> Embedder
    end

    Q --> FP
    FP -->|No| MQ
    FP -->|Yes| VS
    MQ --> VS
    VS --> RK
    RK --> TH
    TH -->|Yes| LLM
    TH -->|No| REF
    LLM --> Answer["Answer + Citations"]

    U --> Loader
    Embedder --> VS

    style QueryLayer fill:#fff4e6,stroke:#d9480f,stroke-width:2px
    style RetrievalEngine fill:#e6f7ff,stroke:#0050b3,stroke-width:2px
    style UserInterface fill:#f9f9f9,stroke:#333
    style IngestionPipeline fill:#f3f0ff,stroke:#5f3dc4,stroke-width:2px
```

> This diagram shows the core retrieval and ingestion pipeline. The three ATTS-RAG security layers that wrap it are detailed in the next section.

---

## 🛡️ ATTS-RAG Security Architecture

ATTS-RAG wraps the pipeline above in three security layers. All three layers are implemented and tested. The trust metadata (`source_tier`, `anomaly_score`, content hashes) written at ingestion time is actively used at query time for authorization, integrity, and provenance verification.

```mermaid
flowchart TD
    UQ["User Query"] --> L1

    subgraph L1["🔴 Layer 1 — Adaptive Threat Intelligence Gate"]
        L1a["Prompt Injection /<br/>Jailbreak Detection"] --> L1b["Dynamic Risk Score"] --> L1c["Sanitize / Reject"]
    end

    L1 --> Retrieval["Hybrid Retrieval"]
    Retrieval --> L2

    subgraph L2["🔵 Layer 2 — Multi-Factor Trust Gate"]
        L2a["Trust Score<br/>Relevance + Reputation"] --> L2b["Provenance +<br/>Poisoning Check"] --> L2c["Re-rank by Trust"]
    end

    L2 --> Gen["Contract-Constrained<br/>Generation"]
    Gen --> L3

    subgraph L3["🟢 Layer 3 — Evidence-to-Answer Verification Gate"]
        L3a["Fast-Fail Safety<br/>+ Schema Validation"] --> L3b["Direct NLI Claim<br/>Verification"] --> L3c["Failure Router<br/>+ Reconstruction"]
    end

    L3 --> Final["Verified Answer + Citations"]

    style L1 fill:#fff0f0,stroke:#c92a2a,stroke-width:2px
    style L2 fill:#e6f7ff,stroke:#0050b3,stroke-width:2px
    style L3 fill:#ebfbee,stroke:#2b8a3e,stroke-width:2px
```

| Layer | What it does | Status | Where |
| :--- | :--- | :---: | :--- |
| **Knowledge Ingestion** | OCR + chunking, local embedding, content-hash registry, hybrid vector/BM25 index. | ✅ Implemented | [`loader.py`](app/rag/loader.py), [`chunker.py`](app/rag/chunker.py), [`vectorstore.py`](app/rag/vectorstore.py) |
| **Ingestion Poisoning Scan** | Regex threat feed + LLM judge + embedding-outlier scoring on every uploaded chunk; quarantines rather than indexes suspicious content. | ✅ Implemented | [`ingestion_guard.py`](app/rag/ingestion_guard.py), [`injection_patterns.yaml`](config/injection_patterns.yaml) |
| **Layer 1 — Threat Intelligence Gate** | Screens user queries for injection/jailbreak attempts with multi-detector ensemble, dynamic risk scoring, and per-session escalation. | ✅ Implemented | [`threat_gate.py`](app/rag/threat_gate.py) |
| **Layer 2 — Multi-Factor Trust Gate** | Discrete decision tree using source tier, anomaly score, content-hash integrity, tenant authorization, and poisoning re-check to tier and filter retrieved evidence. | ✅ Implemented | [`trust_gate.py`](app/rag/trust_gate.py) |
| **Layer 3 — Evidence-to-Answer Verification** | Contract-constrained generation, fast-fail safety scanning, direct NLI claim verification (O(N)), evidence-gap-aware failure routing, and verified answer reconstruction. | ✅ Implemented | [`app/rag/layer3/`](app/rag/layer3/) |
| **Offline Improvement Flywheel** | Telemetry → Failure/Canary Review → Root-Cause Analysis → Dataset Curation → Model/Retrieval/Prompt Improvements → Regression Benchmark → Versioned Deployment. | ✅ Architecture Defined | [`Layer3_Evidence_To_Answer_Gate.md`](docs/Layer3_Evidence_To_Answer_Gate.md) |

> [!NOTE]
> All three ATTS-RAG security layers are implemented and tested. The ingestion and retrieval engine is stable and in daily use. See [`implementation_plan.md`](implementation_plan.md) for architectural details and [`docs/Layer3_Evidence_To_Answer_Gate.md`](docs/Layer3_Evidence_To_Answer_Gate.md) for the Layer 3 specification.

---

## � Quick Start (5 Minutes)

### 1. Prerequisite Checklist
*   [ ] **Python 3.11+** installed.
*   [ ] **Ollama** installed and running.
*   [ ] Run `ollama pull llama3`.

### 2. Setup
```powershell
# Clone & Navigate
git clone https://github.com/SakthiQ/ask-my-docs.git
cd ask-my-docs

# Environment Initialization
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Launch
```powershell
# Start the Backend
uvicorn app.main:app --reload

# Start the Frontend (New!)
streamlit run frontend/streamlit_app.py
```

### 4. Admin uploads (optional)
Uploads default to the `unknown` source tier. To assign a higher tier (`official`, `verified_internal`, `approved_external`) or upload a new version of an existing document, copy `.env.example` to `.env`, set `ADMIN_TOKEN` to a long random value, and enter it in the sidebar's **Admin token** field.

Every upload is scanned for hidden instructions before indexing. Suspicious chunks are quarantined, never indexed, and listed in the Document Library for review.

To add trust metadata to documents ingested before the scan existed, stop the backend and rebuild the index (the old one is kept as a timestamped backup):
```powershell
python scripts/reingest_corpus.py --rebuild --tier official
```

To check that ingestion is complete and the vector index, keyword index and registry agree:
```powershell
python scripts/check_index.py
```
---

---

## Try It Locally (one-line)

If you already have Python and `ollama` installed, run this quick sequence in PowerShell to boot the backend and interactive UI (run in two shells):

```powershell
# Create venv, install deps, start backend and frontend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
streamlit run frontend/streamlit_app.py
```

---

## 🧩 Interactive Demo

- **Quick run:** Start the backend and then run the Streamlit UI:

```powershell
# In one shell: start backend
uvicorn app.main:app --reload

# In another shell: start the interactive frontend
streamlit run frontend/streamlit_app.py
```

- **What the demo does:**
  - Upload PDFs/DOCX/TXT from the sidebar to ingest into the local registry.
  - Ask questions in the chat; the app shows answers, citations, and an optional reasoning trace.
  - Manage the indexed document library (view metadata or delete entries).

- **Tips:**
  - Ensure the backend `API_URL` in `frontend/streamlit_app.py` matches your backend host (default `http://127.0.0.1:8000`).
  - For local LLM inference make sure Ollama is running and models are pulled.

## 🚀 Innovations & Next Ideas

Here are small, high-impact ideas to make the project more interactive and innovative. Pick any to prototype next.

- **Real-time Document Stream:** Stream uploads and incremental chunking to show ingestion progress.
- **Explainability Panel:** Visualize re-ranking scores, sentence-level provenance, and why an answer was selected.
- **Model Switcher:** Allow switching LLMs (local Ollama models) from the UI for A/B testing.
- **Collaborative Sessions:** Real-time shared chat rooms for teams to jointly query a document set.
- **Browser Extension:** Quick-context selection in the browser that queries the local RAG instance.
- **Plugin Marketplace:** Small connectors to common enterprise sources (Confluence, SharePoint, Google Drive) with privacy-first sync.
- **Interactive Citation Explorer:** Click a source to preview the page and highlighted excerpt inline.

---

## 📚 API Guide

<details>
<summary>📂 <b>View Endpoints & Examples</b></summary>

### Upload Document
`POST /upload`
```bash
curl -X POST "http://127.0.0.1:8000/upload" -F "file=@/path/to/Policy.pdf"
```

### Ask AI
`POST /query`
```json
{
  "question": "What is the annual leave policy?"
}
```

</details>

---

## �️ Technology Stack

*   **Orchestration**: LangChain, FastAPI
*   **Vector Database**: ChromaDB (Atomic Persistence)
*   **Search**: Hybrid (Vector + BM25Okapi)
*   **Re-ranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
*   **Embeddings**: HuggingFace `all-MiniLM-L6-v2`
*   **Logging**: Loguru & Tenacity (Retry Logic)

---

## 🖼️ Screenshots

_Preview (placeholders — add screenshots to `assets/` and update paths):_

<details>
<summary>Open screenshots</summary>

![UI preview](assets/screenshot-1.png)

</details>

---

## 🤝 Contributing

We welcome contributions. Small, focused PRs are easiest to review. Please follow these guidelines:

- Create a branch from `main` named `feature/your-feature` or `fix/issue-number`.
- Keep PRs small and include a short description of changes and motivation.
- Run tests (if present) and linters before opening a PR.

See `CONTRIBUTING.md` (if added) for more details.

---

## ⚙️ How to Update This Project on GitHub

Use these PowerShell commands from the repository root to create a branch, commit your changes, and open a Pull Request. Replace the branch name and messages as appropriate.

```powershell
# 1) Sync local main
git checkout main
git pull origin main

# 2) Create a feature branch
git checkout -b feature/your-short-name

# 3) Stage and commit
git add README.md frontend/streamlit_app.py
git commit -m "docs(ui): improve README layout and add screenshots placeholder"

# 4) Push and create PR
git push -u origin feature/your-short-name
# Optional: create PR with GitHub CLI
gh pr create --fill --base main --head feature/your-short-name
```

Use `--force-with-lease` only when rewriting history and you understand the consequences.

---

## 📈 Roadmap

- [x] **Phase 1-3**: Basic RAG, FastAPI, and Advanced Retrieval.
- [x] **Phase 4**: Hybrid Search & Cross-Encoder Reranking.
- [x] **Phase 5**: Agentic Research Loops & HyDE Routing (later removed: they added three LLM calls per query without improving answers).
- [x] **Phase 6**: Multimodal Support (Images/Tables in PDFs).
- [x] **Phase 7**: Evaluation Framework (local LLM-as-judge).
- [x] **Phase 8a**: Ingestion-time poisoning scan (threat feed + LLM judge + embedding outlier detection, quarantine on match).
- [x] **Phase 8b — ATTS-RAG Layer 1**: Query-time threat intelligence gate (injection/jailbreak screening, dynamic risk score, session history).
- [x] **Phase 8c — ATTS-RAG Layer 2**: Trust-weighted re-ranking and provenance verification at query time, using metadata already captured at ingestion.
- [x] **Phase 8d — ATTS-RAG Layer 3**: Evidence-to-answer verification gate (contract-constrained generation, direct NLI claim verification, evidence-gap-aware failure routing, verified answer reconstruction).
- [x] **Phase 8e — Offline Improvement Flywheel**: Telemetry-driven offline process for model, retrieval, and prompt improvements with regression benchmarking and versioned deployment.

See [`implementation_plan.md`](implementation_plan.md) for architectural details and [`docs/Layer3_Evidence_To_Answer_Gate.md`](docs/Layer3_Evidence_To_Answer_Gate.md) for the Layer 3 specification.

---

<div align="center">
  <p>Built with ❤️ for Privacy and Performance.</p>
  <a href="#table-of-contents">Back to Top</a>
</div>

