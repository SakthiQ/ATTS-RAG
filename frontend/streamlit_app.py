import streamlit as st
import requests
import threading
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="ATTS-RAG — Adaptive Threat-Intelligence RAG",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:8000"
SOURCE_TIER_OPTIONS = {
    "🥇 Official Policy / Verified Standard": {
        "tier": "official",
        "weight": 1.00,
        "badge": "🟢 Highest Security Tier",
        "desc": "Signed enterprise policies, compliance standards, and verified core specs."
    },
    "🥈 Verified Internal Knowledge / Wiki": {
        "tier": "verified_internal",
        "weight": 0.85,
        "badge": "🔵 Internal Trusted Tier",
        "desc": "Team wikis, internal engineering docs, and verified internal notes."
    },
    "🥉 Approved Third-Party & Vendor Docs": {
        "tier": "approved_external",
        "weight": 0.70,
        "badge": "🟡 Partner Approved Tier",
        "desc": "Vetted vendor documentation and trusted external reference guides."
    },
    "⚪ General / Unvetted Document": {
        "tier": "unknown",
        "weight": 0.40,
        "badge": "⚪ Standard Upload Tier",
        "desc": "Default tier for standard user uploads and unverified imports."
    },
    "🚫 Untrusted / Raw External Import": {
        "tier": "untrusted",
        "weight": 0.00,
        "badge": "🔴 Zero Trust Tier",
        "desc": "Raw web content or files flagged for potential prompt injection risk."
    }
}

# Session State Setup
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# ChatGPT Dark Color Tokens
bg_app = "#0d0d0d"
bg_sidebar = "#171717"
bg_card = "#212121"
bg_subtle = "#2f2f2f"
border_card = "#2f2f2f"
text_primary = "#ececf1"
text_secondary = "#c5c5d2"
text_muted = "#8e8ea0"
accent_orange = "#f97316"
accent_orange_bg = "rgba(249, 115, 22, 0.15)"

st.markdown(f"""
<style>
    /* Global App Container (Pure ChatGPT Dark) */
    .stApp {{
        background-color: {bg_app} !important;
        color: {text_primary} !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}

    /* Streamlit Native Top Header & Footer Transparent Override */
    header[data-testid="stHeader"], 
    div[data-testid="stHeader"], 
    footer,
    div[data-testid="stBottom"], 
    div[data-testid="stBottom"] > div {{
        background-color: transparent !important;
        background: transparent !important;
    }}

    /* Sidebar Styling (ChatGPT Desktop Style) */
    section[data-testid="stSidebar"] {{
        background-color: {bg_sidebar} !important;
        border-right: 1px solid #1f1f1f !important;
    }}
    
    section[data-testid="stSidebar"] * {{
        color: {text_primary} !important;
    }}

    /* Expander Header */
    div[data-testid="stExpander"] summary,
    details summary {{
        background-color: {bg_card} !important;
        color: {text_primary} !important;
        border: 1px solid {border_card} !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
    }}

    div[data-testid="stExpander"] summary *,
    details summary * {{
        color: {text_primary} !important;
        font-weight: 600 !important;
    }}

    div[data-testid="stExpander"] {{
        background-color: {bg_card} !important;
        border: 1px solid {border_card} !important;
        border-radius: 10px !important;
    }}

    /* Main Conversation Centered Workspace */
    .main .block-container {{
        max-width: 840px !important;
        padding-top: 1rem !important;
        padding-bottom: 6rem !important;
    }}

    /* Typography */
    h1, h2, h3, h4, h5, h6, p, span, label, div {{
        color: {text_primary};
    }}

    /* ChatGPT Mode Switcher Pill at Top Center */
    .top-mode-container {{
        display: flex;
        justify-content: center;
        margin-top: 10px;
        margin-bottom: 40px;
    }}

    .top-mode-pill {{
        background: #171717;
        border: 1px solid #2f2f2f;
        border-radius: 20px;
        padding: 4px;
        display: inline-flex;
        gap: 4px;
    }}

    .mode-tab {{
        padding: 6px 16px;
        border-radius: 16px;
        font-size: 13px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease;
    }}

    .mode-tab.active {{
        background: #212121;
        color: #ffffff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }}

    .mode-tab.inactive {{
        color: #8e8ea0;
    }}

    /* Hero Section: "Hey, Sakthi. Ready to dive in?" */
    .hero-container {{
        text-align: center;
        padding: 40px 20px 20px 20px;
    }}

    .hero-greeting {{
        font-size: 30px;
        font-weight: 600;
        color: #ececf1;
        margin-bottom: 24px;
        letter-spacing: -0.4px;
    }}

    /* ChatGPT Pill Input Bar */
    div[data-testid="stChatInput"],
    div[data-testid="stChatInput"] > div,
    div[data-testid="stChatInput"] [data-baseweb="base-input"],
    div[data-testid="stChatInput"] [data-baseweb="textarea"] {{
        background-color: {bg_card} !important;
        border: 1px solid {border_card} !important;
        border-radius: 28px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
    }}

    div[data-testid="stChatInput"] textarea {{
        color: {text_primary} !important;
        -webkit-text-fill-color: {text_primary} !important;
        background-color: transparent !important;
        font-weight: 400 !important;
        font-size: 15px !important;
        padding-left: 12px !important;
    }}

    div[data-testid="stChatInput"] textarea::placeholder {{
        color: {text_muted} !important;
        -webkit-text-fill-color: {text_muted} !important;
        opacity: 0.8 !important;
    }}

    /* ChatGPT Signature Orange Submit Button */
    div[data-testid="stChatInput"] button {{
        background: {accent_orange} !important;
        background-image: linear-gradient(135deg, #f97316 0%, #ea580c 100%) !important;
        border-radius: 50% !important;
        width: 36px !important;
        height: 36px !important;
        box-shadow: 0 2px 10px rgba(249, 115, 22, 0.4) !important;
        transition: all 0.2s ease !important;
    }}

    div[data-testid="stChatInput"] button:hover {{
        transform: scale(1.05) !important;
        box-shadow: 0 4px 16px rgba(249, 115, 22, 0.6) !important;
    }}

    div[data-testid="stChatInput"] button * {{
        color: #ffffff !important;
    }}

    /* Chat Messages */
    div[data-testid="stChatMessage"] {{
        background-color: {bg_card} !important;
        border: 1px solid {border_card} !important;
        border-radius: 14px !important;
        padding: 16px 18px !important;
        margin-bottom: 14px !important;
    }}

    /* User Message Style */
    .user-msg-bubble {{
        background-color: #2f2f2f !important;
        color: #ececf1 !important;
        padding: 12px 18px !important;
        border-radius: 18px !important;
        display: inline-block !important;
        font-size: 14.5px !important;
    }}

    /* Security Gate Badges */
    .gate-pills-wrapper {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 14px;
        padding-top: 12px;
        border-top: 1px solid #2f2f2f;
    }}

    .gate-pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11.5px;
        font-weight: 600;
    }}

    .gate-pill.pass {{
        background: rgba(16, 185, 129, 0.12) !important;
        color: #34d399 !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
    }}

    .gate-pill.fail {{
        background: rgba(239, 68, 68, 0.12) !important;
        color: #f87171 !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
    }}

    /* Sidebar Buttons */
    button, div[data-testid="stButton"] > button, .stButton button {{
        background-color: {bg_card} !important;
        color: {text_primary} !important;
        border: 1px solid {border_card} !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }}
    
    button *, div[data-testid="stButton"] > button *, .stButton button * {{
        color: {text_primary} !important;
    }}

    button:hover, div[data-testid="stButton"] > button:hover {{
        border-color: #424242 !important;
        background-color: #2a2a2a !important;
    }}

    button[kind="primary"], button[data-testid="baseButton-primary"] {{
        background: {accent_orange} !important;
        color: #ffffff !important;
        border: none !important;
    }}
    
    button[kind="primary"] *, button[data-testid="baseButton-primary"] * {{
        color: #ffffff !important;
    }}

    /* Inputs */
    input, select, div[data-baseweb="select"] {{
        background-color: {bg_subtle} !important;
        color: {text_primary} !important;
        border-color: {border_card} !important;
        border-radius: 8px !important;
    }}

    /* Sidebar User Profile Footer */
    .sidebar-user-profile {{
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 12px;
        border-radius: 10px;
        background: #212121;
        margin-top: 20px;
    }}
    .user-avatar-icon {{
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: #f97316;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 14px;
    }}
</style>
""", unsafe_allow_html=True)


# Helper function to check system health
@st.cache_data(ttl=10, show_spinner=False)
def check_system_health():
    status = {"api": False, "ollama": False, "vector_db": False, "chunk_count": 0}
    try:
        r = requests.get(f"{API_URL}/health", timeout=10)
        if r.status_code == 200:
            data = r.json()
            details = data.get("details", {})
            status["api"] = True
            ollama = details.get("ollama", {})
            vdb = details.get("vector_db", {})
            status["ollama"] = ollama.get("ok", False) if isinstance(ollama, dict) else bool(ollama)
            status["vector_db"] = vdb.get("ok", False) if isinstance(vdb, dict) else bool(vdb)
            
            vdetail = vdb.get("detail", "") if isinstance(vdb, dict) else ""
            if "chunks indexed" in vdetail:
                try:
                    status["chunk_count"] = int(vdetail.split()[0])
                except Exception:
                    pass
    except Exception:
        if "last_health" in st.session_state:
            return st.session_state["last_health"]
        pass

    if status["api"]:
        st.session_state["last_health"] = status
    elif "last_health" in st.session_state:
        return st.session_state["last_health"]

    return status


# Helper function to get document registry
def get_documents():
    try:
        r = requests.get(f"{API_URL}/documents", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


# Helper function to format clean short filenames for UI cards
def shorten_filename(filename, max_len=24):
    if not filename:
        return "Document"
    parts = filename.rsplit(".", 1)
    name = parts[0]
    ext = f".{parts[1]}" if len(parts) > 1 else ""
    name_clean = name.replace("_", " ").strip()
    if len(name_clean) + len(ext) > max_len:
        avail = max_len - len(ext) - 3
        if avail < 4:
            avail = 4
        return f"{name_clean[:avail]}...{ext}"
    return f"{name_clean}{ext}"


# Helper function to generate document-aware suggested questions
def get_suggested_questions(docs_registry):
    questions = []
    
    if docs_registry:
        filenames = [meta.get("filename", "Document") for meta in docs_registry.values() if meta.get("filename")]
        if filenames:
            f1 = filenames[0]
            f1_short = shorten_filename(f1, 24)
            questions.append((f"📄 Key takeaways from {f1_short}?", f"What are the main key points and takeaways in {f1}?"))
            questions.append((f"🔍 Compliance & security rules in {f1_short}?", f"What policy and security requirements are defined in {f1}?"))
            
            if len(filenames) > 1:
                f2 = filenames[1]
                f2_short = shorten_filename(f2, 24)
                questions.append((f"📑 Overview of topics in {f2_short}?", f"Summarize the main topics in {f2}"))
            else:
                questions.append((f"🛡️ How does Layer 2 verify evidence from {f1_short}?", f"How does Layer 2 verify evidence from {f1}?"))
                
            questions.append(("📊 Summarize key insights across all documents", "Summarize key information across all uploaded documents"))
    
    defaults = [
        ("🔒 What security policies protect enterprise data?", "What security policies protect enterprise data?"),
        ("🛡️ How does Layer 1 block prompt injection?", "How does Layer 1 block prompt injection?"),
        ("⚖️ How does Layer 2 filter untrusted evidence?", "How does Layer 2 filter untrusted evidence?"),
        ("🟢 How does Layer 3 verify NLI claims?", "How does Layer 3 verify claims against hallucinations?")
    ]
    
    for label, qtext in defaults:
        if len(questions) < 4 and not any(q[1] == qtext for q in questions):
            questions.append((label, qtext))
            
    return questions[:4]



health = check_system_health()
docs_registry = get_documents()
total_docs = len(docs_registry)
total_chunks = health.get("chunk_count", 0)

# =========================================================================
# LEFT SIDEBAR: CHATGPT DESKTOP STYLE NAVIGATION
# =========================================================================
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:16px;">
        <div style="font-size:18px; font-weight:700; letter-spacing:-0.4px;">ATTS-RAG</div>
        <div style="font-size:14px; opacity:0.6;">🔍</div>
    </div>
    """, unsafe_allow_html=True)

    # New Chat Button
    if st.button("📝 New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)

    # Upload Document Expander
    with st.expander("📤 Upload Document", expanded=False):
        uploaded_file = st.file_uploader(
            "Upload Document File",
            type=["pdf", "docx", "txt", "md"],
            help="Supported: PDF, DOCX, TXT, MD",
            key=f"file_uploader_{st.session_state.uploader_key}"
        )
        
        selected_tier_label = st.selectbox(
            "Source Trust Tier",
            options=list(SOURCE_TIER_OPTIONS.keys()),
            index=3  # Default to 'General / Unvetted'
        )
        tier_info = SOURCE_TIER_OPTIONS[selected_tier_label]
        source_tier = tier_info["tier"]

        st.markdown(f"""
        <div style="background:#1a1c22; border:1px solid #2f323e; border-radius:8px; padding:10px; margin: 4px 0 10px 0;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:11px; font-weight:700; color:#ececf1;">{tier_info['badge']}</span>
            </div>
            <div style="font-size:11px; color:#8e8ea0; margin-top:4px; line-height:1.3;">{tier_info['desc']}</div>
        </div>
        """, unsafe_allow_html=True)
        document_id = st.text_input("Document ID (Optional)", placeholder="e.g. policy_v1")
        admin_token = st.text_input("Admin Token", type="password", placeholder="Required for verified tiers")

        if st.button("🚀 Upload & Index Document", type="primary", use_container_width=True):
            if uploaded_file is None:
                st.warning("Select a file first.")
            else:
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    form = {"source_tier": source_tier, "document_id": document_id.strip()}
                    headers = {"X-Admin-Token": admin_token} if admin_token else {}
                    with st.spinner("Ingesting & Scanning..."):
                        resp = requests.post(f"{API_URL}/upload", files=files, data=form, headers=headers, timeout=40)
                    if resp.status_code == 200:
                        st.session_state.uploader_key += 1
                        st.success(f"Ingested {uploaded_file.name}!")
                        st.rerun()
                    else:
                        st.error(f"Failed ({resp.status_code}): {resp.text}")
                except Exception as exc:
                    st.error(f"Error: {exc}")

    # Document Library & Pinned Items
    with st.expander(f"📚 Document Library ({total_docs})", expanded=True):
        if not docs_registry:
            st.caption("No documents ingested yet.")
        else:
            for doc_hash, meta in docs_registry.items():
                filename = meta.get("filename", "Document")
                tier = meta.get("source_tier", "unknown")
                chunks = meta.get("chunk_count", 0)
                quarantined = meta.get("quarantined", [])
                
                ext = filename.split(".")[-1].lower() if "." in filename else ""
                doc_icon = "📄" if ext == "pdf" else "📝" if ext == "docx" else "📑" if ext == "md" else "📜"

                st.markdown(f"""
                <div style="font-size:12.5px; font-weight:600; margin-top:8px;">
                    {doc_icon} {filename}
                </div>
                <div style="font-size:11px; opacity:0.75; margin-bottom:6px;">
                    Tier: <b>{tier}</b> · Chunks: <b>{chunks}</b><br/>
                    Status: <span style="color:#10b981;">{'⚠️ Quarantined' if quarantined else '✓ Verified Clean'}</span>
                </div>
                """, unsafe_allow_html=True)
                
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("Preview", key=f"sb_p_{doc_hash}", use_container_width=True):
                        try:
                            presp = requests.get(f"{API_URL}/documents/{doc_hash}/preview", timeout=5)
                            if presp.status_code == 200:
                                st.code(presp.json().get("preview", ""))
                        except Exception:
                            st.error("Error")
                with b2:
                    if st.button("Delete", key=f"sb_d_{doc_hash}", use_container_width=True):
                        try:
                            dresp = requests.delete(f"{API_URL}/documents/{doc_hash}", timeout=5)
                            if dresp.status_code == 200:
                                st.rerun()
                        except Exception:
                            st.error("Error")



    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:11.5px; opacity:0.8;">
        <b>System Telemetry:</b><br/>
        • Backend API: <span style="color:{'#10b981' if health['api'] else '#ef4444'}">{'🟢 Online' if health['api'] else '🔴 Offline'}</span><br/>
        • Ollama Llama-3: <span style="color:{'#10b981' if health['ollama'] else '#ef4444'}">{'🟢 Loaded' if health['ollama'] else '🔴 Unreachable'}</span><br/>
        • Chroma Vector DB: <span style="color:{'#10b981' if health['vector_db'] else '#ef4444'}">{'🟢 Ready' if health['vector_db'] else '🔴 Error'}</span> ({total_chunks} Chunks)
    </div>
    """, unsafe_allow_html=True)

    # ChatGPT User Profile Footer
    st.markdown("""
    <div class="sidebar-user-profile">
        <div class="user-avatar-icon">S</div>
        <div>
            <div style="font-size:13px; font-weight:600;">Sakthi Narayan</div>
            <div style="font-size:11px; color:#8e8ea0;">Enterprise Tier</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================================
# MAIN CHAT WORKSPACE
# =========================================================================

# Top Status Indicator
if health["api"]:
    st.markdown("<div style='text-align:right; font-size:12px; color:#10b981; font-weight:600;'>🟢 Index Online</div>", unsafe_allow_html=True)
else:
    st.markdown("<div style='text-align:right; font-size:12px; color:#ef4444; font-weight:600;'>🔴 Backend Offline</div>", unsafe_allow_html=True)

# Hero Screen if conversation is empty
if not st.session_state.messages:
    st.markdown("""
    <div class="hero-container">
        <div class="hero-greeting">Hey, Sakthi. Ready to dive in?</div>
    </div>
    """, unsafe_allow_html=True)

# Render Chat History
for msg in st.session_state.messages:
    role = msg.get("role", "user")
    avatar_icon = "👤" if role == "user" else "🛡️"
    with st.chat_message(role, avatar=avatar_icon):
        if role == "user":
            st.markdown(f'<div class="user-msg-bubble">{msg.get("content", "")}</div>', unsafe_allow_html=True)
        else:
            st.markdown(msg.get("content", ""))
        
        if role == "assistant":
            l1 = msg.get("threat_gate", {})
            l2 = msg.get("layer2_gate", {})
            l3 = msg.get("layer3_gate", {})
            
            l1_pass = l1.get("allowed", True)
            l2_pass = l2.get("allowed", True)
            l3_pass = (l3.get("decision") == "PASS") if l3 else True

            st.markdown(
                f"""
                <div class="gate-pills-wrapper">
                    <span class="gate-pill {'pass' if l1_pass else 'fail'}">
                        🛡️ L1 Threat Gate: <b>{'PASS' if l1_pass else 'BLOCK'}</b>
                    </span>
                    <span class="gate-pill {'pass' if l2_pass else 'fail'}">
                        ⚖️ L2 Trust Gate: <b>{'VERIFIED' if l2_pass else 'REJECT'}</b>
                    </span>
                    <span class="gate-pill {'pass' if l3_pass else 'fail'}">
                        🔒 L3 Output Gate: <b>{'PASS' if l3_pass else 'REJECT'}</b>
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Layer 3 & Telemetry Claim Inspector Expander
            l3_telemetry = l3.get("telemetry") if l3 else None
            claim_results = l3_telemetry.get("claim_results", []) if l3_telemetry else []
            if claim_results or l1 or l2:
                with st.expander("📊 Detailed Security & Claim Verification Explorer"):
                    st.markdown("#### 🔒 Security Gate Telemetry")
                    l1_t = l1.get("execution_time_ms", 0.0) if l1 else 0.0
                    l2_t = l2.get("execution_time_ms", 0.0) if l2 else 0.0
                    l3_t = l3.get("execution_time_ms", 0.0) if l3 else 0.0
                    
                    st.markdown(f"""
                    <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:10px; margin-bottom:12px;">
                        <div style="background:#111216; border:1px solid #2f323e; padding:10px; border-radius:8px; text-align:center;">
                            <div style="font-size:11px; color:#8e8ea0;">Layer 1 Risk</div>
                            <div style="font-size:16px; font-weight:700; color:{'#10b981' if l1_pass else '#ef4444'};">{l1.get('final_risk', 0.0):.3f}</div>
                            <div style="font-size:10px; color:#6b7280;">{l1_t:.1f} ms</div>
                        </div>
                        <div style="background:#111216; border:1px solid #2f323e; padding:10px; border-radius:8px; text-align:center;">
                            <div style="font-size:11px; color:#8e8ea0;">Layer 2 Status</div>
                            <div style="font-size:14px; font-weight:700; color:{'#10b981' if l2_pass else '#ef4444'};">{l2.get('status', 'N/A')}</div>
                            <div style="font-size:10px; color:#6b7280;">{l2_t:.1f} ms</div>
                        </div>
                        <div style="background:#111216; border:1px solid #2f323e; padding:10px; border-radius:8px; text-align:center;">
                            <div style="font-size:11px; color:#8e8ea0;">Layer 3 Verdict</div>
                            <div style="font-size:14px; font-weight:700; color:{'#10b981' if l3_pass else '#ef4444'};">{l3.get('decision', 'N/A')}</div>
                            <div style="font-size:10px; color:#6b7280;">{l3_t:.1f} ms</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if claim_results:
                        st.markdown("#### ✅ Per-Claim NLI Verification Results")
                        for cres in claim_results:
                            cid = cres.get("claim_id", "C?")
                            ctext = cres.get("text", "")
                            cstatus = cres.get("status", "UNKNOWN")
                            conf = cres.get("confidence", 0.0) * 100
                            creason = cres.get("reason", "")
                            
                            badge_color = "#10b981" if cstatus == "ENTAILMENT" else ("#f59e0b" if cstatus == "INSUFFICIENT" else "#ef4444")
                            
                            st.markdown(f"""
                            <div style="background:#1a1c22; border-left:3px solid {badge_color}; border-radius:6px; padding:10px 14px; margin-bottom:8px;">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <span style="font-weight:700; font-size:12.5px; color:#ececf1;">Claim [{cid}]</span>
                                    <span style="background:{badge_color}22; color:{badge_color}; border:1px solid {badge_color}44; padding:2px 8px; border-radius:12px; font-size:11px; font-weight:700;">
                                        {cstatus} ({conf:.1f}%)
                                    </span>
                                </div>
                                <div style="font-size:12.5px; color:#d1d5db; margin-top:4px;">"{ctext}"</div>
                                <div style="font-size:11px; color:#8e8ea0; margin-top:4px;"><i>Reason: {creason}</i></div>
                            </div>
                            """, unsafe_allow_html=True)

            citation_details = msg.get("citation_details", [])
            citations = msg.get("citations", [])
            if citation_details or citations:
                count = len(citation_details) if citation_details else len(citations)
                with st.expander(f"🔍 Interactive Citation Inspector ({count} Verified Chunks)"):
                    if citation_details:
                        for idx, cdet in enumerate(citation_details):
                            fname = cdet.get("filename", "Document")
                            tier = cdet.get("source_tier", "unknown")
                            rel = cdet.get("relevance_score", 0.0)
                            weight = cdet.get("trust_weight", 1.0)
                            snippet = cdet.get("snippet", "")
                            cid = cdet.get("id", f"chunk_{idx}")
                            
                            st.markdown(f"""
                            <div style="background:#1a1c22; border:1px solid #2f323e; border-radius:10px; padding:12px; margin-bottom:10px;">
                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                                    <span style="font-weight:700; font-size:13px; color:#ececf1;">📄 {fname}</span>
                                    <span style="background:rgba(16,163,127,0.15); color:#10a37f; border:1px solid rgba(16,163,127,0.3); padding:2px 8px; border-radius:12px; font-size:11px; font-weight:600;">Tier: {tier}</span>
                                </div>
                                <div style="font-size:11px; color:#8e8ea0; margin-bottom:8px;">
                                    Chunk ID: <code>{cid}</code> · Relevance Score: <b style="color:#10b981;">{rel}</b>
                                </div>
                                <div style="font-size:12.5px; color:#c5c5d2; background:#111216; padding:10px; border-radius:6px; border-left:3px solid #10a37f; line-height:1.4;">
                                    "{snippet}..."
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        for cid in citations:
                            st.markdown(f"• `{cid}`")

# Handle execution if the last message is from user and awaiting assistant response
if st.session_state.messages and st.session_state.messages[-1].get("role") == "user":
    last_user_query = st.session_state.messages[-1].get("content")
    with st.chat_message("assistant", avatar="🛡️"):
        status_placeholder = st.empty()
        status_placeholder.markdown("""
        <div style="background:#1a1c22; border:1px solid #2f323e; border-left:3px solid #f97316; padding:14px 18px; border-radius:10px; margin-bottom:12px; color:#ececf1;">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
                <span style="display:inline-block; width:16px; height:16px; border:2.5px solid #f97316; border-top-color:transparent; border-radius:50%; animation:spin 1s linear infinite;"></span>
                <span style="font-weight:700; font-size:14px; color:#f97316;">ATTS-RAG 3-Layer Security Pipeline Active</span>
            </div>
            <div style="font-size:12.5px; color:#c5c5d2; line-height:1.6; padding-left:26px;">
                • 🔴 <b>Layer 1 Threat Gate</b>: Screening query for prompt injection & jailbreak risks...<br/>
                • 🔍 <b>Hybrid Search</b>: Searching ChromaDB Vector Store + BM25 Keyword Index...<br/>
                • ⚡ <b>Cross-Encoder Rerank</b>: Scoring top relevant document passages...<br/>
                • 🔵 <b>Layer 2 Trust Gate</b>: Verifying source trust tiers & document provenance...<br/>
                • 🧠 <b>LLM Synthesis</b>: Generating contract-constrained response...<br/>
                • 🟢 <b>Layer 3 Verification</b>: Running NLI anti-hallucination claim check...
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            resp = requests.post(f"{API_URL}/query", json={"question": last_user_query}, timeout=300)
            status_placeholder.empty()
            
            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer", "No answer returned.")
                citations = data.get("citations", [])
                citation_details = data.get("citation_details", [])
                
                msg_obj = {
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "citation_details": citation_details,
                    "threat_gate": data.get("threat_gate"),
                    "layer2_gate": data.get("layer2_gate"),
                    "layer3_gate": data.get("layer3_gate")
                }
                st.session_state.messages.append(msg_obj)
                st.rerun()
            else:
                st.error(f"API Error ({resp.status_code}): {resp.text}")
        except Exception as exc:
            status_placeholder.empty()
            st.error(f"Connection failure: {exc}")

# Floating Bottom Chat Input (ChatGPT Capsule Bar)
prompt = st.chat_input("Ask anything...")

# Dynamic Suggested Question Cards (Rendered beneath input area when conversation is empty)
if not st.session_state.messages:
    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    suggested_qs = get_suggested_questions(docs_registry)
    
    r1_col1, r1_col2 = st.columns(2)
    r2_col1, r2_col2 = st.columns(2)
    cols = [r1_col1, r1_col2, r2_col1, r2_col2]
    
    for idx, (label, qtext) in enumerate(suggested_qs):
        with cols[idx]:
            if st.button(label, use_container_width=True, key=f"sq_btn_{idx}"):
                st.session_state.pending_question = qtext
                st.rerun()

active_prompt = prompt or st.session_state.pop("pending_question", None)

if active_prompt:
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    st.rerun()
