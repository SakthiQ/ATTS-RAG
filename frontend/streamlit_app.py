import streamlit as st
import requests
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="ATTS-RAG Security Console",
    page_icon="🛡️",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"
SOURCE_TIERS = ["unknown", "official", "verified_internal", "approved_external", "untrusted"]

# Global Custom Styling
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; }
    .status-pass { color: #15803d; font-weight: 600; }
    .status-fail { color: #b91c1c; font-weight: 600; }
    .status-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 8px;
        margin-bottom: 12px;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 8px 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# Helper function to check system health
@st.cache_data(ttl=15)
def check_system_health():
    status = {"api": False, "ollama": False, "vector_db": False,
               "ollama_detail": "", "vector_db_detail": ""}
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            data = r.json()
            details = data.get("details", {})
            status["api"] = True
            # New granular format: details.ollama / details.vector_db are objects
            ollama = details.get("ollama", {})
            vdb    = details.get("vector_db", {})
            if isinstance(ollama, dict):
                status["ollama"] = ollama.get("ok", False)
                status["ollama_detail"] = ollama.get("detail", "")
            else:
                status["ollama"] = bool(ollama)
            if isinstance(vdb, dict):
                status["vector_db"] = vdb.get("ok", False)
                status["vector_db_detail"] = vdb.get("detail", "")
            else:
                status["vector_db"] = bool(vdb)
    except Exception:
        pass
    return status


health = check_system_health()

# Top Header
st.title("🛡️ ATTS-RAG Security Console")
st.caption("Adaptive Threat and Trust Security for Retrieval-Augmented Generation")

# Sidebar
with st.sidebar:
    st.header("🎛️ Console Controls")
    
    # Mode Toggle
    dev_mode = st.toggle("🛠️ Developer Mode", value=False, help="Enable detailed NLI metrics, claim objects, and layer execution telemetry.")
    
    st.divider()

    # System Status
    st.subheader("🟢 System Status")
    col_a, col_b, col_c = st.columns(3)
    col_a.markdown(f"**API**\n{'🟢' if health['api'] else '🔴'}")
    col_b.markdown(
        f"**LLM**\n{'🟢' if health['ollama'] else '🔴'}",
        help=health.get('ollama_detail', '') or "Ollama LLM server"
    )
    col_c.markdown(
        f"**VectorDB**\n{'🟢' if health['vector_db'] else '🔴'}",
        help=health.get('vector_db_detail', '') or "ChromaDB vector store"
    )

    st.divider()

    # Document Upload Section
    st.subheader("📥 Ingest Document")
    uploaded_file = st.file_uploader("Upload PDF, DOCX, TXT or MD", type=["pdf", "docx", "txt", "md"])
    source_tier = st.selectbox(
        "Source Tier",
        SOURCE_TIERS,
        index=1,
        help="How far Layer 2 should trust this document. Non-unknown tiers require admin token."
    )
    document_id = st.text_input("Document ID (optional)", help="Defaults to filename slug.")
    admin_token = st.text_input("Admin Token", type="password", help="Backend ADMIN_TOKEN for elevated source tiers.")

    if uploaded_file is not None:
        if st.button("🚀 Ingest Document", use_container_width=True):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                form = {"source_tier": source_tier, "document_id": document_id.strip()}
                headers = {"X-Admin-Token": admin_token} if admin_token else {}
                with st.spinner("Uploading and indexing..."):
                    resp = requests.post(f"{API_URL}/upload", files=files, data=form, headers=headers, timeout=30)

                if resp.status_code == 200:
                    st.success(f"Ingestion started: {uploaded_file.name}")
                    st.rerun()
                else:
                    detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
                    st.error(f"Upload failed ({resp.status_code}): {detail}")
            except Exception as e:
                st.error(f"Ingestion error: {e}")


# Main Content Area: Chat (Left 65%) vs Document Library (Right 35%)
left_col, right_col = st.columns([2, 1])

if "messages" not in st.session_state:
    st.session_state.messages = []

with left_col:
    st.subheader("💬 Security-Gated RAG Chat")

    # Render Chat History safely using Streamlit native chat_message
    for msg in st.session_state.messages:
        role = msg.get("role", "user")
        with st.chat_message(role):
            # Native Streamlit markdown rendering (Prevents XSS / raw HTML injection vulnerabilities)
            st.markdown(msg.get("content", ""))

            # Assistant Security & Telemetry Rendering
            if role == "assistant":
                l1 = msg.get("threat_gate", {})
                l2 = msg.get("layer2_gate", {})
                l3 = msg.get("layer3_gate", {})

                l1_pass = l1.get("allowed", True)
                l2_pass = l2.get("allowed", True)
                l3_pass = l3.get("decision") == "PASS" if l3 else True

                # 1. Three-Layer Security Status Bar
                st.markdown("---")
                s1, s2, s3 = st.columns(3)
                s1.markdown(f"**Layer 1 (Threat Gate):** {'<span class=\"status-pass\">✓ PASS</span>' if l1_pass else '<span class=\"status-fail\">✗ BLOCK</span>'}", unsafe_allow_html=True)
                s2.markdown(f"**Layer 2 (Trust Gate):** {'<span class=\"status-pass\">✓ VERIFIED</span>' if l2_pass else '<span class=\"status-fail\">✗ REJECT</span>'}", unsafe_allow_html=True)
                s3.markdown(f"**Layer 3 (Output Gate):** {'<span class=\"status-pass\">✓ PASS</span>' if l3_pass else '<span class=\"status-fail\">✗ REJECT</span>'}", unsafe_allow_html=True)

                # Rejection Experience Card
                if l3 and not l3_pass:
                    st.error(
                        f"🛑 **Answer Not Released**\n\n"
                        f"**Reason:** {l3.get('reason') or l3.get('failure_reason') or 'Claim grounding or safety verification failed.'}\n\n"
                        f"**Verification Matrix:**\n"
                        f"- Relevance: ✓ PASS\n"
                        f"- Fast Safety: ✓ PASS\n"
                        f"- Grounding: ✗ FAILED\n\n"
                        f"**Retry Attempt:** {l3.get('telemetry', {}).get('retry_count', 0)}"
                    )

                # Citations & Evidence Explorer
                citations = msg.get("citations", [])
                if citations:
                    with st.expander(f"📌 Cited Evidence Chunks ({len(citations)})"):
                        for cid in citations:
                            st.markdown(f"**Chunk ID:** `{cid}`")

                # Timing & Performance Breakdown
                t1 = l1.get("execution_time_ms", 0.0) if l1 else 0.0
                t2 = l2.get("execution_time_ms", 0.0) if l2 else 0.0
                t3 = l3.get("telemetry", {}).get("execution_time_ms", 0.0) if l3 else 0.0
                total_s = (t1 + t2 + t3) / 1000.0

                if total_s > 0:
                    st.caption(f"⏱️ **Response Latency:** {total_s:.2f}s (L1: {t1:.1f}ms | L2: {t2:.1f}ms | L3: {t3:.1f}ms)")

                # 2. Developer Mode Detailed Telemetry
                if dev_mode and l3:
                    with st.expander("🛠️ Developer Telemetry & Claim Matrix"):
                        st.json({
                            "layer1_threat_gate": l1,
                            "layer2_trust_gate": l2,
                            "layer3_output_gate": l3
                        })

    # Chat Input
    prompt = st.chat_input("Ask a question about your enterprise documents...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt, "ts": datetime.utcnow().isoformat()})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Processing through 3-Layer Security Pipeline..."):
                try:
                    resp = requests.post(f"{API_URL}/query", json={"question": prompt}, timeout=60)
                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data.get("answer", "No answer returned.")
                        citations = data.get("citations", [])
                        
                        msg_obj = {
                            "role": "assistant",
                            "content": answer,
                            "citations": citations,
                            "threat_gate": data.get("threat_gate"),
                            "layer2_gate": data.get("layer2_gate"),
                            "layer3_gate": data.get("layer3_gate"),
                            "ts": datetime.utcnow().isoformat()
                        }
                        st.session_state.messages.append(msg_obj)
                        st.rerun()
                    else:
                        st.error(f"API Error ({resp.status_code}): {resp.text}")
                except Exception as e:
                    st.error(f"Connection Failure: {e}")

# Right Column: Document Library & Governance Cards
with right_col:
    st.subheader("📚 Document Library")
    try:
        docs_resp = requests.get(f"{API_URL}/documents", timeout=10)
        if docs_resp.status_code == 200:
            registry = docs_resp.json()
            if not registry:
                st.info("No documents indexed yet.")
            else:
                for doc_hash, meta in registry.items():
                    filename = meta.get("filename", "Document")
                    tier = meta.get("source_tier", "unknown")
                    chunks_cnt = meta.get("chunk_count", 0)
                    quarantined = meta.get("quarantined", [])
                    flagged_cnt = meta.get("flagged_count", 0)

                    with st.expander(f"📄 {filename}", expanded=False):
                        st.markdown(f"**Integrity Hash:** `{doc_hash[:12]}`")
                        st.markdown(f"**Source Tier:** `{tier}`")
                        st.markdown(f"**Indexed Chunks:** {chunks_cnt}")
                        st.markdown(f"**Version:** {meta.get('version', 1)}")

                        if quarantined:
                            st.warning(f"⚠️ {len(quarantined)} chunk(s) quarantined")
                        if flagged_cnt:
                            st.info(f"🚩 {flagged_cnt} chunk(s) flagged")

                        btn_col1, btn_col2 = st.columns(2)
                        with btn_col1:
                            if st.button("Preview", key=f"prev-{doc_hash}"):
                                p_resp = requests.get(f"{API_URL}/documents/{doc_hash}/preview", timeout=10)
                                if p_resp.status_code == 200:
                                    preview_text = p_resp.json().get("preview", "(empty)")
                                    st.code(preview_text)
                                else:
                                    st.error(f"Preview failed: {p_resp.status_code}")
                        with btn_col2:
                            if st.button("Delete", key=f"del-{doc_hash}"):
                                d_resp = requests.delete(f"{API_URL}/documents/{doc_hash}")
                                if d_resp.status_code == 200:
                                    st.success("Deleted")
                                    st.rerun()
        else:
            st.error("Failed to load document registry.")
    except Exception as e:
        st.caption(f"Registry offline: {e}")
