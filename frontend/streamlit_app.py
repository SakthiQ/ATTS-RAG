import streamlit as st
import requests
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Ask My Documents",
    page_icon="📚",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"
SOURCE_TIERS = ["unknown", "official", "verified_internal", "approved_external", "untrusted"]

st.title("📚 Ask My Documents")
st.markdown("### Privacy-first local RAG platform powered by Ollama.")

# Basic styles for nicer UI
BASE_CSS = """
<style>
  .stApp { background: #f7f9fb; }
  .chat-bubble { background: #ffffff; padding: 12px; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.06); margin-bottom: 8px; }
  .chat-user { background: linear-gradient(90deg,#e6f2ff,#ffffff); }
  .meta { color: #6c757d; font-size:12px }
  .stButton>button { border-radius: 8px; height:38px }
</style>
"""

st.markdown(BASE_CSS, unsafe_allow_html=True)

# Sidebar controls: upload + settings
with st.sidebar:
    st.header("📥 Ingest & Settings")

    # Settings
    st.subheader("Settings")
    st.caption("Local model: llama3 (via Ollama). All inference runs on this machine.")

    st.divider()
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF, DOCX, TXT or MD file", type=["pdf", "docx", "txt", "md"])
    source_tier = st.selectbox(
        "Source tier",
        SOURCE_TIERS,
        help="How far Layer 2 should trust this document. Any tier other than 'unknown' needs the admin token.",
    )
    document_id = st.text_input(
        "Document ID (optional)",
        help="Defaults to the filename. Reuse an existing ID to upload a new version of that document (admin only).",
    )
    admin_token = st.text_input("Admin token", type="password", help="Must match ADMIN_TOKEN in the backend's .env file.")

    if uploaded_file is not None:
        if st.button("🚀 Ingest Document"):
            status_text = st.empty()
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                form = {"source_tier": source_tier, "document_id": document_id.strip()}
                headers = {"X-Admin-Token": admin_token} if admin_token else {}
                with st.spinner("Uploading..."):
                    resp = requests.post(f"{API_URL}/upload", files=files, data=form, headers=headers, timeout=30)
                if resp.status_code == 200:
                    status_text.success(f"Ingestion started: {uploaded_file.name}")
                    st.rerun()
                else:
                    try:
                        detail = resp.json().get("detail", resp.text)
                    except ValueError:
                        detail = resp.text
                    status_text.error(f"Ingest failed ({resp.status_code}): {detail}")
            except Exception as e:
                status_text.error(f"Upload error: {e}")

    st.divider()
    st.markdown("---")
    st.markdown("Need a demo or help? Open an issue on the repo.")

# Main layout: chat (left) and document panel (right)
left, right = st.columns([3, 1])

if "messages" not in st.session_state:
    st.session_state.messages = []

with left:
    st.subheader("Conversation")

    # Display chat
    for message in st.session_state.messages:
        role = message.get("role", "user")
        timestamp = message.get("ts")
        ts_text = f"{timestamp}" if timestamp else ""
        with st.container():
            classes = "chat-bubble"
            if role == "user":
                classes += " chat-user"
            st.markdown(f"<div class='{classes}'><div class='meta'>{role} · {ts_text}</div>\n\n{message.get('content')}</div>", unsafe_allow_html=True)

    # Chat input
    prompt = st.chat_input("Ask a question about your documents...")
    if prompt:
        # user message
        st.session_state.messages.append({"role": "user", "content": prompt, "ts": datetime.utcnow().isoformat()})

        # assistant placeholder
        with st.spinner("Thinking..."):
            try:
                payload = {"question": prompt}
                resp = requests.post(f"{API_URL}/query", json=payload, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("answer", "No answer returned.")
                    citations = data.get("citations", [])
                    reasoning_log = data.get("reasoning_log", [])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                        "reasoning_log": reasoning_log,
                        "ts": datetime.utcnow().isoformat()
                    })

                    # re-render to show new message immediately
                    st.rerun()
                else:
                    st.error(f"API error: {resp.status_code} - {resp.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

with right:
    st.subheader("📚 Document Library")
    try:
        docs_resp = requests.get(f"{API_URL}/documents", timeout=10)
        if docs_resp.status_code == 200:
            registry = docs_resp.json()
            if not registry:
                st.info("No documents indexed yet.")
            else:
                for doc_hash, meta in registry.items():
                    with st.expander(f"{meta.get('filename')}"):
                        st.write(f"**Hash:** `{doc_hash[:12]}`")
                        st.write(f"**Source tier:** {meta.get('source_tier', 'unknown (legacy)')} · **Version:** {meta.get('version', 1)}")
                        st.write(f"**Chunks indexed:** {meta.get('chunk_count')}")
                        quarantined = meta.get("quarantined", [])
                        if quarantined:
                            st.warning(f"{len(quarantined)} chunk(s) quarantined by the poisoning scan")
                            for q in quarantined:
                                st.caption(f"Score {q.get('anomaly_score')}: {q.get('notes')}")
                        if meta.get("flagged_count"):
                            st.info(f"{meta['flagged_count']} chunk(s) flagged for Layer 2 review")
                        if st.button("Preview text", key=f"preview-{doc_hash}"):
                            preview_resp = requests.get(f"{API_URL}/documents/{doc_hash}/preview", timeout=10)
                            if preview_resp.status_code == 200:
                                st.code(preview_resp.text)
                            else:
                                st.warning("No preview available.")
                        if st.button("Delete", key=f"del-{doc_hash}"):
                            del_resp = requests.delete(f"{API_URL}/documents/{doc_hash}")
                            if del_resp.status_code == 200:
                                st.success("Document removed.")
                                st.rerun()
                            else:
                                st.error("Delete failed.")
        else:
            st.error("Failed to load registry.")
    except Exception as e:
        st.warning(f"Registry unavailable: {e}")

    st.markdown("---")
    st.caption("Tip: set the OLLAMA_MODEL environment variable on the backend to change the local LLM used for queries.")
