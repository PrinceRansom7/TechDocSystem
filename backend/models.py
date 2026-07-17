"""
Pydantic request/response models for all FastAPI endpoints.
"""

from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# ── /query ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000, description="User's technical question")
    response_mode: Literal["concise", "detailed"] = Field("concise", description="Response verbosity")
    enable_web_search: bool = Field(False, description="Activate web-search fallback node")
    session_id: Optional[str] = Field(None, description="Session ID for conversation memory")
    force_rewrite: bool = Field(False, description="Force an aggressive semantic rewrite")
    expand_query: bool = Field(False, description="Expand query with synonyms to improve retrieval")


class SourceDoc(BaseModel):
    document_id: str
    source: str                      # filename or URL
    section_label: Optional[str] = None
    parent_path: Optional[str] = None
    semantic_role: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    chunk_text: Optional[str] = None  # shown only when "show chunks" is toggled


class QueryResponse(BaseModel):
    answer: str
    query_type: str                   # TECHNICAL_QUERY | GENERAL_CHAT
    sources: list[SourceDoc] = []
    discarded_sources: list[SourceDoc] = []
    web_sources: list[dict[str, str]] = []
    retries_used: int = 0
    fallback_used: bool = False
    hallucination_status: str = "unchecked"


# ── /ingest (URL-based) ───────────────────────────────────────────────────────

class IngestURLRequest(BaseModel):
    urls: list[str] = Field(..., min_length=1, description="List of URLs to fetch and ingest")
    title_hint: Optional[str] = Field(None, description="Optional document title override")


class IngestResponse(BaseModel):
    ingested: int
    skipped: int
    details: list[dict[str, Any]] = []


# ── /documents ────────────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    document_id: str
    source: str
    title: Optional[str] = None
    document_type: Optional[str] = None
    chunk_count: int
    page_count: Optional[int] = None


class DocumentListResponse(BaseModel):
    total_chunks: int
    documents: list[DocumentInfo]


# ── /feedback ─────────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: Literal["up", "down"] = Field(..., description="Thumbs up or down")
    comment: Optional[str] = Field(None, max_length=1000)


class FeedbackResponse(BaseModel):
    message: str


# ── /hallucination_check ──────────────────────────────────────────────────────

class HallucinationCheckRequest(BaseModel):
    answer: str = Field(..., description="The generated answer to verify")
    context_chunks: list[str] = Field(..., description="List of context chunks to verify against")


class HallucinationCheckResponse(BaseModel):
    status: Literal["pass", "fail"] = Field(..., description="Pass or fail verdict")


# ── /health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = Field("ok")
    version: str = Field("1.0.0")
