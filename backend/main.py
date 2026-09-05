"""
FastAPI application — TechDocSystem RAG API.

Endpoints:
  POST /query        → run the LangGraph RAG pipeline
  POST /ingest/url   → ingest documents from URLs
  POST /ingest/file  → upload & ingest local files
  GET  /documents    → list indexed documents
  POST /feedback     → submit thumbs-up / thumbs-down rating
  GET  /health       → health check
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from backend.config import get_settings
from backend.models import (
    DocumentListResponse,
    FeedbackRequest,
    FeedbackResponse,
    IngestResponse,
    IngestURLRequest,
    QueryRequest,
    QueryResponse,
    SourceDoc,
    HallucinationCheckRequest,
    HallucinationCheckResponse,
    HealthResponse,
)
from backend.ingest import ingest_file, ingest_url, list_documents
from backend.graph import run_rag_pipeline, run_rag_pipeline_stream

# ── Logging ───────────────────────────────────────────────────────────────────
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="TechDocSystem — RAG Technical Documentation Assistant",
    description=(
        "Self-corrective LangGraph RAG system for answering questions about technical documents. "
        "Supports PDF, Markdown, plain text, and URL ingestion."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (for conversation memory)
_sessions: dict[str, list[dict]] = {}

# Simple feedback store (in production use a DB)
_feedback_log: list[dict] = []


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "model": settings.groq_model}


# ── POST /query ───────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query(req: QueryRequest):
    """Submit a technical question to the LangGraph RAG pipeline."""
    session_id = req.session_id or str(uuid.uuid4())
    history = _sessions.get(session_id, [])

    try:
        result = run_rag_pipeline(
            question=req.question,
            settings=settings,
            response_mode=req.response_mode,
            enable_web_search=req.enable_web_search,
            session_history=history,
        )
    except Exception as e:
        logger.exception("Pipeline error: %s", e)
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {e}")

    # Update session history
    history.append({"role": "user", "content": req.question})
    history.append({"role": "assistant", "content": result["answer"]})
    _sessions[session_id] = history[-20:]  # keep last 10 turns

    sources = [SourceDoc(**s) for s in result.get("sources", [])]
    discarded_sources = [SourceDoc(**s) for s in result.get("discarded_sources", [])]
    return QueryResponse(
        answer=result["answer"],
        query_type=result["query_type"],
        sources=sources,
        discarded_sources=discarded_sources,
        web_sources=result["web_sources"],
        retries_used=result["retries_used"],
        fallback_used=result["fallback_used"],
        hallucination_status=result.get("hallucination_status", "unchecked"),
    )


# ── POST /query/stream ─────────────────────────────────────────────────────────

@app.post("/query/stream", tags=["RAG"])
def query_stream(req: QueryRequest):
    """Submit a technical question to the LangGraph RAG pipeline and stream updates."""
    session_id = req.session_id or str(uuid.uuid4())
    history = _sessions.get(session_id, [])

    def event_generator():
        try:
            generator = run_rag_pipeline_stream(
                question=req.question,
                settings=settings,
                response_mode=req.response_mode,
                enable_web_search=req.enable_web_search,
                session_history=history,
                force_rewrite=req.force_rewrite,
                expand_query=req.expand_query,
            )
            final_answer = ""
            for event_str in generator:
                yield event_str
                # Try to extract the final answer to update history
                if '"node": "final_result"' in event_str:
                    import json
                    try:
                        data = json.loads(event_str.replace("data: ", "").strip())
                        final_answer = data["result"]["answer"]
                    except:
                        pass
            
            if final_answer:
                history.append({"role": "user", "content": req.question})
                history.append({"role": "assistant", "content": final_answer})
                _sessions[session_id] = history[-20:]
                
        except Exception as e:
            logger.exception("Pipeline streaming error: %s", e)
            import json
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── POST /ingest/url ──────────────────────────────────────────────────────────

@app.post("/ingest/url", response_model=IngestResponse, tags=["Ingestion"])
def ingest_urls(req: IngestURLRequest):
    """Ingest one or more technical documentation URLs into the vector store."""
    ingested = 0
    skipped = 0
    details = []
    for url in req.urls:
        r = ingest_url(url, settings)
        details.append(r)
        if r.get("error"):
            logger.warning("Ingest error for %s: %s", url, r["error"])
        elif r.get("skipped"):
            skipped += 1
        else:
            ingested += r.get("chunks_added", 0)

    return IngestResponse(ingested=ingested, skipped=skipped, details=details)


# ── POST /ingest/file ─────────────────────────────────────────────────────────

@app.post("/ingest/file", response_model=IngestResponse, tags=["Ingestion"])
async def upload_and_ingest(
    files: list[UploadFile] = File(...),
    library: str = Form(default=""),
    version: str = Form(default=""),
):
    """Upload PDF, Markdown, or plain text files and ingest them."""
    upload_dir = settings.upload_path
    upload_dir.mkdir(parents=True, exist_ok=True)

    ingested = 0
    skipped = 0
    details = []
    for file in files:
        suffix = Path(file.filename or "doc").suffix.lower()
        if suffix not in (".pdf", ".md", ".mdx", ".txt", ".rst"):
            details.append({"source": file.filename, "error": f"Unsupported file type: {suffix}"})
            continue
        save_path = upload_dir / (file.filename or f"upload_{uuid.uuid4()}{suffix}")
        content = await file.read()
        save_path.write_bytes(content)
        r = ingest_file(
            save_path,
            settings,
            library=library or None,
            version=version or None,
        )
        details.append(r)
        if r.get("error"):
            logger.warning("Ingest error for %s: %s", file.filename, r["error"])
        elif r.get("skipped"):
            skipped += 1
        else:
            ingested += r.get("chunks_added", 0)

    return IngestResponse(ingested=ingested, skipped=skipped, details=details)


# ── GET /documents ────────────────────────────────────────────────────────────

@app.get("/documents", response_model=DocumentListResponse, tags=["Ingestion"])
def documents():
    """List all documents currently indexed in the vector store."""
    docs = list_documents(settings)
    total_chunks = sum(d.get("chunk_count", 0) for d in docs)
    from backend.models import DocumentInfo
    return DocumentListResponse(
        total_chunks=total_chunks,
        documents=[DocumentInfo(**d) for d in docs],
    )


# ── POST /feedback ────────────────────────────────────────────────────────────

@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
def feedback(req: FeedbackRequest):
    """Record user feedback (thumbs up / down) for a given answer."""
    entry = req.model_dump()
    _feedback_log.append(entry)
    return FeedbackResponse(message="Feedback received")


# ── GET /health ───────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok", version="1.0.0")


# ── DELETE /sessions/{session_id} ─────────────────────────────────────────────

@app.delete("/sessions/{session_id}", tags=["System"])
def clear_session(session_id: str):
    """Clear conversation memory for a session."""
    if session_id in _sessions:
        del _sessions[session_id]
        return {"message": f"Session {session_id} cleared"}
    return {"message": "Session not found"}


# ── POST /hallucination_check ─────────────────────────────────────────────────

@app.post("/hallucination_check", response_model=HallucinationCheckResponse, tags=["RAG"])
def ad_hoc_hallucination_check(req: HallucinationCheckRequest):
    """Run an isolated Self-RAG hallucination check on a generated answer."""
    from backend.graph import _make_llm
    
    if not req.context_chunks:
        return HallucinationCheckResponse(status="pass")
        
    context_str = "\n\n".join(req.context_chunks)
    
    check_prompt = f"""You are a strict factual grounding evaluator (Self-RAG ISSUP).

Your task: determine whether the GENERATED ANSWER is fully supported by the PROVIDED CONTEXT.

Rules:
- Answer PASS if every factual claim in the answer can be traced to the context.
- Answer FAIL if the answer contains ANY claim, detail, or fact not present in the context (hallucination).
- Ignore phrasing and style differences; focus only on factual accuracy.

PROVIDED CONTEXT:
{context_str}

GENERATED ANSWER:
{req.answer}

Respond with ONLY one word: PASS or FAIL."""

    try:
        llm = _make_llm(settings)
        verdict = llm.invoke(check_prompt).content.strip().upper()
        status = "pass" if verdict == "PASS" else "fail"
        return HallucinationCheckResponse(status=status)
    except Exception as e:
        logger.exception("Hallucination check failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
