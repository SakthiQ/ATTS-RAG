# ATTS-RAG Frontend Architecture & Design Report

**Platform:** ATTS-RAG (Adaptive Threat and Trust Security for Retrieval-Augmented Generation)  
**Interface:** Streamlit Security Console (`frontend/streamlit_app.py`)  
**Target Audience:** Enterprise Security Analysts, System Administrators & AI Developers  
**Date:** October 18, 2026  

---

## 1. Executive Summary

The **ATTS-RAG Security Console** turns an ordinary RAG chat interface into an **enterprise security control panel**. Instead of treating the RAG engine as a black box, the console makes all three security boundaries—**Input Threat Screening (Layer 1)**, **Retrieval Trust & Provenance (Layer 2)**, and **Output Grounding & Safety Gate (Layer 3)**—visually explicit, auditable, and interactive.

---

## 2. Layout

```text
┌─────────────────┬──────────────────────────────────────┬─────────────────────────────┐
│ SIDEBAR         │ WORKSPACE                            │ EVIDENCE PANEL        [ ✕ ] │
│                 │                                      │ (opens on a source click)   │
│ New chat        │ 🎯 Understood as: <intent>           │ ┌ Source passage ┬ Verification ┐
│ Upload document │ Answer, one verified sentence per    │ │ E2 · file p.33              │
│ Document        │ line, each ending in [E#]            │ │ tier · relevance            │
│ library         │ Chart (only if checked)              │ │ Sentences it supports       │
│ (tier, hash,    │ L1 ✓  L2 ✓  L3 ✓                     │ │ Full passage text           │
│  Preview/Delete)│ Sources: [E2 · file p.33] [E3 …]     │ └─────────────────────────────┘
│ System health   │ [ Ask anything…                  ➤ ] │                             │
└─────────────────┴──────────────────────────────────────┴─────────────────────────────┘
```

The evidence panel is closed by default and the workspace stays centred and narrow (840 px). Clicking a **source button** under an answer opens the panel and widens the workspace; **✕** closes it. State is `st.session_state.open_citation = {msg, label}`. Streamlit cannot make words inside markdown clickable, so citations open from the button row rather than from the `[E#]` tags themselves.

---

## 3. Detailed Feature Breakdown

### 🛡️ 1. Three-Layer Security Visibility Bar

Every assistant response renders a status bar displaying the verdict of each security layer:

- **Layer 1 (Adaptive Threat Gate):** `PASS` (green), `BLOCK` (red). `FLAG` is allowed through and shown with its risk score in the explorer.
- **Layer 2 (Multi-Factor Trust Gate):** `VERIFIED` or `REJECT`.
- **Layer 3 (Evidence Verification Gate):** `PASS` or `REJECT`.

### 🎯 2. Question understanding and charts

- The one-line **intent** the model inferred is shown above the answer so users can see how their question was read.
- A **chart** (bar / line / pie, Altair) is drawn only when Layer 3 verified that every label and value occurs in the cited source text; a caption says so.

### 📌 3. Citations and the Evidence Panel

- Each answer sentence ends in `[E#]`; below the answer, one button per used passage (`E2 · file p.33`).
- **Source passage tab:** file, page, trust tier, relevance, the answer sentences that passage supports, and the full passage text.
- **Verification tab:** share of sentences verified, number removed as unsupported, and every sentence with its status (entailment / contradiction / insufficient), confidence and supporting passage.
- A rejected answer shows the closest sources the same way, so users can read the evidence themselves.

### 🛑 4. Structured Rejection Experience

When Layer 3 (or Layer 1/2) rejects a response, the UI replaces the answer with an auditable **Answer Not Released** card:

- Displays the explicit failure classification (e.g., *Claim C2 contradicted evidence* or *High-confidence secret leakage*).
- Offers the closest source passages instead of an answer (Layer 3 does not retry).

### ⏱️ 4. Real-Time Per-Layer Latency Dashboard

Calculates and displays execution latency breakdown across the 3 security layers:

$$\text{Total Latency} = T_{\text{Layer 1}} + T_{\text{Layer 2}} + T_{\text{Layer 3}}$$

- Example: `⏱️ Response Latency: 1.25s (L1: 12.0ms | L2: 110.0ms | L3: 240.0ms)`.

### 🛠️ 5. Dual-View Mode (User View vs. Developer View)

- **User View (Default):** Clean answer text, citation tags, 3-layer status bar, evidence expander, response time.
- **Developer View (Toggle):** Exposes full raw telemetry JSON, claim objects ($C_1..C_n$), NLI classification labels (`ENTAILMENT`, `CONTRADICTION`, `INSUFFICIENT`) and confidence scores.

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

- Model-generated text, document snippets, file names and user input are HTML-escaped wherever the page uses `unsafe_allow_html`.
- Passage and sentence text in the evidence panel is rendered with `st.text`, never as markdown or HTML, so links, images or markup hidden in a document stay inert.
- The API address is configurable with `ATTS_API_URL`.

---

## 4. Summary Matrix

| Metric / Dimension | Baseline Chat UI | ATTS-RAG Security Console |
| :--- | :--- | :--- |
| **Security Layer Visibility** | Layer 3 post-hoc text only | **Real-time 3-Layer Visual Status Bar (L1, L2, L3)** |
| **Evidence Traceability** | Raw text list | **Closable evidence panel: passage text, supported sentences, per-sentence verification** |
| **Rejection Handling** | Generic error string | **Structured Rejection Card & Matrix Checklist** |
| **Telemetry & Debugging** | Hardcoded text | **Developer Mode Toggle with Raw JSON Telemetry** |
| **HTML Security** | Unsafe `st.markdown` | **Escaped values; `st.text` for untrusted passages** |
| **System Health** | None | **Live Health Checks (API, LLM, Vector DB)** |
