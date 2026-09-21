# ATTS-RAG Frontend Architecture & Design Report

**Platform:** ATTS-RAG (Adaptive Threat and Trust Security for Retrieval-Augmented Generation)  
**Interface:** Streamlit Security Console (`frontend/streamlit_app.py`)  
**Target Audience:** Enterprise Security Analysts, System Administrators & AI Developers  
**Date:** September 20, 2026  

---

## 1. Executive Summary

The **ATTS-RAG Security Console** turns an ordinary RAG chat interface into an **enterprise security control panel**. Instead of treating the RAG engine as a black box, the console makes all three security boundaries—**Input Threat Screening (Layer 1)**, **Retrieval Trust & Provenance (Layer 2)**, and **Output Grounding & Safety Gate (Layer 3)**—visually explicit, auditable, and interactive.

---

## 2. Core UI Architectural Enhancements

```text
┌────────────────────────────────────────────────────────────────────────────────┐
│ 🛡️ ATTS-RAG Security Console                             [🛠️ Developer Mode ○] │
├────────────────────────────────────────┬───────────────────────────────────────┤
│ 💬 Security-Gated RAG Chat (65%)       │ 📚 Document Library (35%)             │
│                                        │                                       │
│ User: What is the MFA policy?          │ 📄 it_policy.pdf                      │
│                                        │ ──────────────────────                │
│ Assistant:                              │ Hash: `a8f192b3...`                   │
│ Employees must enable MFA. [E2]        │ Tier: `verified_internal`             │
│                                        │ Chunks: 128                           │
│ ┌────────────────────────────────────┐ │ ⚠️ 2 quarantined                     │
│ │ L1 ✓ PASS | L2 ✓ VERIFIED | L3 ✓ PASS │ │ [Preview] [Delete]                   │
│ └────────────────────────────────────┘ │                                       │
│ ⏱️ Response Latency: 1.25s             │ 📄 handbook.docx                      │
│                                        │                                       │
│ [ Ask your enterprise documents... ] ➤ │                                       │
└────────────────────────────────────────┴───────────────────────────────────────┘
```

---

## 3. Detailed Feature Breakdown

### 🛡️ 1. Three-Layer Security Visibility Bar

Every assistant response renders a 3-column status bar displaying the real-time verdict of each security layer:

- **Layer 1 (Adaptive Threat Gate):** `✓ PASS` (Green) or `✗ BLOCK` (Red) with risk score.
- **Layer 2 (Multi-Factor Trust Gate):** `✓ VERIFIED` (Blue) or `✗ INSUFFICIENT` (Red).
- **Layer 3 (Evidence Verification Gate):** `✓ PASS` (Green) or `✗ REJECT` (Red).

### 📌 2. Citation Tagging & Evidence Explorer

- Claims retain explicit citation tags (e.g., `[chk_101]`, `[E2]`).
- Below verified responses, an **Evidence Explorer** card allows analysts to inspect the exact chunk content, element type, section header, and provenance hash that grounded the statement.

### 🛑 3. Structured Rejection Experience

When Layer 3 (or Layer 1/2) rejects a response, the UI replaces the answer with an auditable **Answer Not Released** card:

- Displays the explicit failure classification (e.g., *Claim C2 contradicted evidence* or *High-confidence secret leakage*).
- Displays the **Verification Check Matrix** (Relevance, Safety, Grounding).
- Displays retry attempt counter ($N \le 1$).

### ⏱️ 4. Real-Time Per-Layer Latency Dashboard

Calculates and displays execution latency breakdown across the 3 security layers:

$$\text{Total Latency} = T_{\text{Layer 1}} + T_{\text{Layer 2}} + T_{\text{Layer 3}}$$

- Example: `⏱️ Response Latency: 1.25s (L1: 12.0ms | L2: 110.0ms | L3: 240.0ms)`.

### 🛠️ 5. Dual-View Mode (User View vs. Developer View)

- **User View (Default):** Clean answer text, citation tags, 3-layer status bar, evidence expander, response time.
- **Developer View (Toggle):** Exposes full raw telemetry JSON, claim objects ($C_1..C_n$), NLI classification labels (`ENTAILMENT`, `CONTRADICTION`, `INSUFFICIENT`), confidence scores, and retry counts.

### 🟢 6. Live System Health Monitoring

The sidebar features live status indicators for core dependencies:

- **API**: FastAPI backend status (`http://127.0.0.1:8000/health`)
- **LLM**: Local Ollama runtime (`llama3`)
- **Vector DB**: ChromaDB & BM25 hybrid index status

### 📚 7. Document Library & Governance Cards

The right column displays rich document governance cards:

- **Integrity Hash**: SHA-256 document content hash.
- **Source Tier**: `official`, `verified_internal`, `approved_external`, `untrusted`.
- **Poisoning Scan Status**: Highlights quarantined chunks flagged by ingestion guards.
- **Action Triggers**: In-line text preview and single-click document deletion.

### 🔒 8. Security Hardening (XSS Prevention)

- Replaced un-sanitized `st.markdown(..., unsafe_allow_html=True)` string interpolation with Streamlit's native `st.chat_message()` component.
- Eliminates potential XSS / code injection vulnerabilities from model-generated text or malicious user inputs.

---

## 4. Summary Matrix

| Metric / Dimension | Baseline Chat UI | ATTS-RAG Security Console |
| :--- | :--- | :--- |
| **Security Layer Visibility** | Layer 3 post-hoc text only | **Real-time 3-Layer Visual Status Bar (L1, L2, L3)** |
| **Evidence Traceability** | Raw text list | **Interactive Evidence Explorer with Chunk UIDs** |
| **Rejection Handling** | Generic error string | **Structured Rejection Card & Matrix Checklist** |
| **Telemetry & Debugging** | Hardcoded text | **Developer Mode Toggle with Raw JSON Telemetry** |
| **HTML Security** | Unsafe `st.markdown` | **Native `st.chat_message` (XSS-safe)** |
| **System Health** | None | **Live Health Checks (API, LLM, Vector DB)** |
