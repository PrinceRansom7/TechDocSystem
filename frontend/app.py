"""
Streamlit frontend — TechDocSystem Technical Documentation Assistant.

Features (inspired by AI_UseCase chatbot):
  - Concise / Detailed response mode toggle
  - Upload documents (PDF, MD, TXT) or ingest URLs
  - Toggle web search fallback
  - Show/hide retrieved chunks with semantic role badges
  - Source traceability (document, section, page)
  - Thumbs up/down feedback per response
  - Conversation memory within session
"""

from __future__ import annotations

import os
import uuid
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TechDocSystem - AI Docs Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global */
body { font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* Headers */
h1 { color: #38bdf8; }
h2, h3 { color: #7dd3fc; }

/* Code blocks */
code { background: #1e293b; color: #93c5fd; border-radius: 4px; padding: 2px 6px; }
pre { background: #1e293b !important; border-radius: 8px; padding: 12px; }

/* Chat messages */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    margin-bottom: 8px;
    padding: 4px 8px;
}

/* Source badges */
.role-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-left: 6px;
}
.role-api_reference { background: #1e3a5f; color: #60a5fa; }
.role-code_example  { background: #1a3a2e; color: #34d399; }
.role-installation  { background: #3b2f1a; color: #fbbf24; }
.role-function_def  { background: #2d1b4e; color: #c084fc; }
.role-class_def     { background: #2d1b4e; color: #e879f9; }
.role-overview      { background: #1e2d3b; color: #7dd3fc; }
.role-other         { background: #1e293b; color: #94a3b8; }

/* Expander */
[data-testid="stExpander"] { border: 1px solid #334155; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Helper functions (defined BEFORE use) ────────────────────────────────────

def _send_feedback(meta: dict, rating: str):
    """POST feedback to the API."""
    try:
        requests.post(
            f"{API_BASE}/feedback",
            json={
                "question": meta.get("question", ""),
                "answer": meta.get("answer", ""),
                "rating": rating,
            },
            timeout=5,
        )
        st.toast("Feedback recorded.")
    except Exception:
        st.toast("Could not send feedback.")


def _render_meta(meta: dict, show_chunks: bool, show_grading: bool = False):
    """Render query type badge, sources, chunk viewer, grading, and action buttons."""
    qtype = meta.get("query_type", "")

    # Query type badge
    qtype_labels = {
        "GENERAL_CHAT": "General Chat",
        "CONCEPTUAL": "Conceptual",
        "HOW_TO": "How-To",
        "TROUBLESHOOTING": "Troubleshooting",
        "API_REFERENCE": "API Reference",
        "COMPARATIVE": "Comparative",
        "DEFINITIONAL": "Definitional",
        "DEBUGGING": "Debugging",
    }
    qtype_label = qtype_labels.get(qtype, qtype.replace("_", " ").title())
    retries = meta.get("retries_used", 0)
    fb_note = "  |  Fallback response" if meta.get("fallback_used") else ""
    web_note = "  |  Web search used" if meta.get("web_sources") else ""
    h_status = meta.get("hallucination_status", "unchecked")
    h_note = f"  |  Hallucination: {h_status.upper()}" if h_status != "unchecked" else ""
    st.caption(f"Query type: {qtype_label}  |  Retries: {retries}{web_note}{fb_note}{h_note}")

    sources = meta.get("sources", [])
    web_sources = meta.get("web_sources", [])

    # Document sources
    if sources:
        with st.expander(f"Sources ({len(sources)})", expanded=False):
            for s in sources:
                role = s.get("semantic_role", "other")
                badge_class = f"role-{role}" if role in [
                    "api_reference", "code_example", "installation",
                    "function_def", "class_def", "overview", "other"
                ] else "role-other"
                badge = f'<span class="role-badge {badge_class}">{role.replace("_", " ")}</span>'
                section = f" § **{s.get('section_label')}**" if s.get("section_label") else ""
                path = f"  \n  *{s.get('parent_path')}*" if s.get("parent_path") else ""
                page = f"  p.{s.get('page_start')}" if s.get("page_start") else ""
                st.markdown(
                    f"`{s.get('source', '')}`{section}{path}{page} {badge}",
                    unsafe_allow_html=True,
                )

    # Web sources
    if web_sources:
        with st.expander(f"Web Sources ({len(web_sources)})", expanded=False):
            for w in web_sources:
                st.markdown(f"[{w.get('source', '')}]({w.get('source', '')})")

    # Retrieved chunks viewer
    if show_chunks and sources:
        with st.expander("Retrieved Chunks", expanded=False):
            for i, s in enumerate(sources):
                chunk_text = s.get("chunk_text", "")
                if not chunk_text:
                    continue
                st.markdown(f"**Chunk {i+1}** — `{s.get('source', '')}`")
                if "```" in chunk_text or chunk_text[:100].startswith("    "):
                    st.code(chunk_text, language=None)
                else:
                    st.text_area(
                        label="",
                        value=chunk_text,
                        height=150,
                        key=f"chunk_{id(meta)}_{i}",
                        disabled=True,
                    )

    # LLM Grading viewer
    if show_grading:
        discarded = meta.get("discarded_sources", [])
        if sources or discarded:
            with st.expander("LLM Document Grading", expanded=False):
                if sources:
                    st.markdown(f"**Kept relevant chunks ({len(sources)})**")
                    for s in sources:
                        st.markdown(f"- `{s.get('source', 'Unknown')}`")
                if discarded:
                    st.markdown(f"**Discarded irrelevant chunks ({len(discarded)})**")
                    for s in discarded:
                        st.markdown(f"- `{s.get('source', 'Unknown')}`")
                        chunk_text = s.get("chunk_text", "")
                        if chunk_text:
                            st.caption(chunk_text[:150].replace('\n', ' ') + "...")

    # Feedback buttons
    col1, col2, _col3 = st.columns([1, 1, 8])
    with col1:
        if st.button("Good", key=f"up_{id(meta)}", help="This answer was helpful"):
            _send_feedback(meta, "up")
    with col2:
        if st.button("Poor", key=f"dn_{id(meta)}", help="This answer was not helpful"):
            _send_feedback(meta, "down")

    # Action Buttons Area
    # Action Buttons Area
    if meta.get("hallucination_status") == "fail":
        st.warning("The generated answer failed the hallucination check but maximum retries were reached.")
        if st.button("Attempt hallucination re-check", key=f"btn_halluc_{id(meta)}"):
            st.session_state.pending_query = meta.get("question")
            st.rerun()
    elif meta.get("hallucination_status") != "fail" and meta.get("answer"):
        if st.button("Run Hallucination Check", key=f"btn_run_halluc_{id(meta)}"):
            with st.spinner("Checking facts against context..."):
                try:
                    chunks = [s.get("chunk_text", "") for s in sources if s.get("chunk_text")]
                    res = requests.post(
                        f"{API_BASE}/hallucination_check",
                        json={"answer": meta.get("answer", ""), "context_chunks": chunks},
                        timeout=30
                    )
                    res.raise_for_status()
                    status = res.json().get("status")
                    if status == "pass":
                        st.toast("✅ Hallucination Check Passed!")
                    else:
                        st.error("❌ Hallucination Check Failed! The answer may contain ungrounded facts.")
                except Exception as e:
                    st.error(f"Check failed: {e}")

    if meta.get("fallback_used") and meta.get("retries_used", 0) > 0:
        st.warning("All automated retries failed to find relevant information.")
        if st.button("Try one more aggressive rewrite", key=f"btn_rewrite_{id(meta)}"):
            st.session_state.pending_query = meta.get("question")
            st.session_state["_force_rewrite"] = True
            st.rerun()

    # Display expand button after every response, highlighted if fallback
    expand_type = "primary" if meta.get("fallback_used") else "secondary"
    if st.button("Expand query to improve retrieval", type=expand_type, key=f"btn_expand_{id(meta)}"):
        st.session_state.pending_query = meta.get("question")
        st.session_state["_expand_query"] = True
        st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## TechDocSystem")
    st.caption("AI-powered Technical Documentation Assistant")
    st.divider()

    # Settings
    st.markdown("### Settings")
    response_mode = st.radio(
        "Response Mode",
        options=["concise", "detailed"],
        format_func=lambda x: "Concise" if x == "concise" else "Detailed",
        horizontal=True,
    )
    enable_web_search = st.toggle(
        "Enable Web Search", value=False,
        help="Falls back to web search if no relevant docs are found."
    )
    show_chunks = st.toggle(
        "Show Retrieved Chunks", value=False,
        help="Display the raw chunks used to generate the answer."
    )
    show_grading = st.toggle(
        "Show LLM Grading", value=False,
        help="See which chunks were kept vs discarded by the LLM grader."
    )

    st.divider()

    # Upload Documents
    st.markdown("### Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, Markdown, or TXT files",
        type=["pdf", "md", "mdx", "txt", "rst"],
        accept_multiple_files=True,
        key="file_uploader",
    )
    lib_name = st.text_input("Library name (optional)", placeholder="e.g. LangChain")
    lib_ver = st.text_input("Version (optional)", placeholder="e.g. 0.2.0")

    if st.button("Ingest Uploaded Files", use_container_width=True, disabled=not uploaded_files):
        with st.spinner("Ingesting files…"):
            files_payload = [
                ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
                for f in uploaded_files
            ]
            data = {}
            if lib_name:
                data["library"] = lib_name
            if lib_ver:
                data["version"] = lib_ver
            try:
                resp = requests.post(
                    f"{API_BASE}/ingest/file",
                    files=files_payload,
                    data=data,
                    timeout=120,
                )
                resp.raise_for_status()
                r = resp.json()
                st.success(f"Ingested {r['ingested']} chunks from {len(uploaded_files)} file(s)")
                if r.get("skipped"):
                    st.info(f"{r['skipped']} file(s) already indexed — skipped.")
            except Exception as e:
                st.error(f"Ingestion error: {e}")

    st.divider()

    # Ingest from URL
    st.markdown("### Ingest from URL")
    url_input = st.text_area(
        "Documentation URLs (one per line)",
        placeholder="https://fastapi.tiangolo.com/tutorial/\nhttps://docs.pydantic.dev/",
        height=100,
    )
    if st.button("Ingest URLs", use_container_width=True, disabled=not url_input.strip()):
        urls = [u.strip() for u in url_input.splitlines() if u.strip()]
        with st.spinner(f"Ingesting {len(urls)} URL(s)…"):
            try:
                resp = requests.post(
                    f"{API_BASE}/ingest/url",
                    json={"urls": urls},
                    timeout=120,
                )
                resp.raise_for_status()
                r = resp.json()
                st.success(f"Ingested {r['ingested']} chunks from {len(urls)} URL(s)")
            except Exception as e:
                st.error(f"URL ingestion error: {e}")

    st.divider()

    # Indexed documents panel
    with st.expander("Indexed Documents", expanded=False):
        if st.button("Refresh", key="refresh_docs"):
            st.session_state["_doc_refresh"] = True
        if st.session_state.get("_doc_refresh"):
            try:
                resp = requests.get(f"{API_BASE}/documents", timeout=10)
                resp.raise_for_status()
                doc_data = resp.json()
                st.caption(f"Total chunks: {doc_data['total_chunks']}")
                for d in doc_data.get("documents", []):
                    st.markdown(
                        f"**{d.get('title') or d.get('source', 'Unknown')}**  \n"
                        f"`{d.get('source', '')}`  |  {d.get('chunk_count', 0)} chunks"
                    )
            except Exception as e:
                st.error(f"Could not load documents: {e}")
            st.session_state["_doc_refresh"] = False

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            try:
                requests.delete(f"{API_BASE}/sessions/{st.session_state.session_id}", timeout=5)
            except:
                pass
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("Show Memory", use_container_width=True):
            st.session_state._show_memory = not st.session_state.get("_show_memory", False)
            
    if st.session_state.get("_show_memory", False):
        with st.expander("Session Memory", expanded=True):
            st.json([{"role": m["role"], "content": m["content"][:100]+"..." if len(m["content"])>100 else m["content"]} for m in st.session_state.messages])


# ── Main chat area ────────────────────────────────────────────────────────────

st.title("TechDocSystem - AI Documentation Assistant")
st.caption(
    "Ask questions about your technical documents. "
    "The system retrieves relevant chunks, grades them, and generates grounded answers with citations."
)

# Initialise session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Replay conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            _render_meta(msg["meta"], show_chunks, show_grading)

# ── Chat input ────────────────────────────────────────────────────────────────

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

user_input = st.chat_input("Ask a technical question about your documents…")

if st.session_state.pending_query:
    prompt = st.session_state.pending_query
    st.session_state.pending_query = None
else:
    prompt = user_input
    st.session_state["_force_rewrite"] = False
    st.session_state["_expand_query"] = False

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_container = st.status("Thinking...", expanded=True)
        try:
            import json
            resp = requests.post(
                f"{API_BASE}/query/stream",
                json={
                    "question": prompt,
                    "response_mode": response_mode,
                    "enable_web_search": enable_web_search,
                    "session_id": st.session_state.session_id,
                    "force_rewrite": st.session_state.get("_force_rewrite", False)
                },
                stream=True,
                timeout=120,
            )
            resp.raise_for_status()
            
            data = None
            for line in resp.iter_lines():
                if line and line.startswith(b"data: "):
                    event_str = line[6:].decode("utf-8")
                    event = json.loads(event_str)
                    
                    if event.get("error"):
                        raise Exception(event["error"])
                        
                    node = event.get("node")
                    update = event.get("update", {})
                    
                    if node == "query_analysis":
                        q = update.get("query", "")
                        r = update.get("retries", 0)
                        if r > 0:
                            status_container.write(f"**Retry {r}**: Rewrote query to: `{q}`")
                        else:
                            if q != prompt:
                                status_container.write(f"**Rewrote query to**: `{q}`")
                            else:
                                status_container.write(f"Query analysis complete.")
                    elif node == "retrieval":
                        docs = update.get("retrieved_docs", [])
                        status_container.write(f"**Retrieved** {len(docs)} chunks from vector store.")
                    elif node == "document_grading":
                        graded = update.get("graded_docs", [])
                        discarded = update.get("discarded_docs", [])
                        status_container.write(f"**Grading**: Kept {len(graded)}, Discarded {len(discarded)}.")
                    elif node == "web_search":
                        webs = update.get("web_results", [])
                        status_container.write(f"**Web Search**: Found {len(webs)} results.")
                    elif node == "hallucination_check":
                        status = update.get("hallucination_status")
                        if status == "fail":
                            status_container.write(f"**Hallucination Check**: Failed! Regenerating answer...")
                        else:
                            status_container.write(f"**Hallucination Check**: Passed.")
                    elif node == "final_result":
                        data = event.get("result")
                        
            status_container.update(label="Done!", state="complete", expanded=False)
            
            if st.session_state.get("_force_rewrite"):
                st.session_state["_force_rewrite"] = False
            if st.session_state.get("_expand_query"):
                st.session_state["_expand_query"] = False

        except Exception as e:
            status_container.update(label="Error", state="error", expanded=True)
            data = {
                "answer": f"Error communicating with the backend: {e}\n\nMake sure the FastAPI server is running on {API_BASE}",
                "query_type": "ERROR",
                "sources": [],
                "discarded_sources": [],
                "web_sources": [],
                "retries_used": 0,
                "fallback_used": True,
                "hallucination_status": "unchecked",
            }

        answer = data.get("answer", "") if data else ""
        st.markdown(answer)
        meta = {**data, "question": prompt, "answer": answer} if data else {}
        if meta:
            _render_meta(meta, show_chunks, show_grading)
            
    if answer and not st.session_state.get("_force_rewrite") and not st.session_state.get("_expand_query"):
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "meta": meta,
        })
