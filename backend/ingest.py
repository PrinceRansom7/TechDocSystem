"""
Full ingestion pipeline: parse → chunk → embed → store.

Supports:
  - Local files: PDF, Markdown (.md/.mdx), plain text
  - Remote URLs: HTML pages, raw Markdown
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from backend.config import Settings
from backend.parsing.extractor import extract_document
from backend.chunking.chunker import chunk_document
from backend.chunking.models import TechChunk

logger = logging.getLogger(__name__)


# ── Vector store + embeddings helpers ────────────────────────────────────────

def _get_vector_store(settings: Settings):
    """Initialise ChromaDB collection."""
    import chromadb
    client = chromadb.PersistentClient(path=str(settings.chroma_persist_path))
    col = client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )
    return client, col


# Module-level model cache so it's loaded once per process
_st_model = None


def _get_st_model(model_name: str):
    """Load (and cache) the sentence-transformers model."""
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model '%s' (downloads once on first run)...", model_name)
        _st_model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded.")
    return _st_model


def _embed_texts(texts: list[str], settings: Settings) -> list[list[float]]:
    """Embed a list of texts using sentence-transformers (local, no API key needed)."""
    model = _get_st_model(settings.embedding_model)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=len(texts) > 10,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine-similarity friendly
    )
    return embeddings.tolist()


# ── Metadata helpers ──────────────────────────────────────────────────────────

def _flat_meta(chunk: TechChunk) -> dict:
    """Return a Chroma-compatible metadata dict (str/int/float/bool only)."""
    d = chunk.to_dict()
    flat: dict = {}
    for k, v in d.items():
        if k == "text":
            continue  # stored separately as document
        if isinstance(v, (str, int, float, bool)):
            flat[k] = v
        elif v is None:
            pass
    return flat


# ── Core ingest function ──────────────────────────────────────────────────────

def ingest_source(
    source: str,
    settings: Settings,
    library: Optional[str] = None,
    version: Optional[str] = None,
    is_url: bool = False,
    force_rechunk: bool = False,
) -> dict:
    """
    Parse, chunk, embed, and store a single document source.

    Returns:
        dict with keys: source, document_id, chunks_added, skipped, error
    """
    result = {"source": source, "document_id": None, "chunks_added": 0, "skipped": 0, "error": None}

    # 1. Extract
    extracted = extract_document(source, is_url=is_url)
    if extracted is None:
        result["error"] = f"Failed to extract content from {source}"
        logger.warning(result["error"])
        return result

    document_id = TechChunk.document_id_from_source(source)
    result["document_id"] = document_id

    # 2. Connect to ChromaDB
    try:
        _, col = _get_vector_store(settings)
    except Exception as e:
        result["error"] = f"ChromaDB connection failed: {e}"
        return result

    # 3. Skip if already ingested (unless forced)
    if not force_rechunk:
        existing = col.get(where={"document_id": document_id}, limit=1)
        if existing and existing.get("ids"):
            logger.info("Document %s already ingested (%s). Skipping.", document_id, source)
            result["skipped"] = 1
            return result

    # 4. Chunk
    doc_meta = {
        "source": source,
        "title": extracted.title,
        "library": library,
        "version": version,
    }
    chunks: list[TechChunk] = chunk_document(
        extracted,
        document_id=document_id,
        doc_meta=doc_meta,
        strategy=settings.chunk_strategy,
        max_chunk_chars=settings.max_chunk_chars,
        min_chunk_chars=settings.min_chunk_chars,
        overlap_chars=settings.overlap_chars,
    )
    if not chunks:
        result["error"] = "No chunks produced"
        return result

    logger.info("Produced %d chunks from %s", len(chunks), source)

    # 5. Embed
    texts = [c.text for c in chunks]
    try:
        embeddings = _embed_texts(texts, settings)
    except Exception as e:
        result["error"] = f"Embedding failed: {e}"
        return result

    # 6. Upsert into ChromaDB in batches
    BATCH = 100
    ids = [c.chunk_id for c in chunks]
    metadatas = [_flat_meta(c) for c in chunks]
    for i in range(0, len(ids), BATCH):
        col.upsert(
            ids=ids[i: i + BATCH],
            embeddings=embeddings[i: i + BATCH],
            documents=texts[i: i + BATCH],
            metadatas=metadatas[i: i + BATCH],
        )

    result["chunks_added"] = len(chunks)
    logger.info("Ingested %d chunks for document %s", len(chunks), document_id)
    return result


def ingest_file(
    file_path: str | Path,
    settings: Settings,
    library: Optional[str] = None,
    version: Optional[str] = None,
) -> dict:
    """Ingest a local file."""
    return ingest_source(str(file_path), settings, library=library, version=version, is_url=False)


def ingest_url(
    url: str,
    settings: Settings,
    library: Optional[str] = None,
    version: Optional[str] = None,
) -> dict:
    """Ingest a remote URL."""
    return ingest_source(url, settings, library=library, version=version, is_url=True)


# ── Document listing ──────────────────────────────────────────────────────────

def list_documents(settings: Settings) -> list[dict]:
    """
    Return a deduplicated list of indexed documents with metadata.
    """
    try:
        _, col = _get_vector_store(settings)
    except Exception as e:
        logger.error("ChromaDB error: %s", e)
        return []

    total = col.count()
    if total == 0:
        return []

    # Fetch all metadata in batches
    all_meta: list[dict] = []
    offset = 0
    BATCH = 500
    while offset < total:
        result = col.get(limit=BATCH, offset=offset, include=["metadatas"])
        all_meta.extend(result.get("metadatas") or [])
        offset += BATCH

    # Deduplicate by document_id
    docs: dict[str, dict] = {}
    for m in all_meta:
        did = m.get("document_id", "unknown")
        if did not in docs:
            docs[did] = {
                "document_id": did,
                "source": m.get("source", ""),
                "title": m.get("title"),
                "document_type": m.get("source_type"),
                "chunk_count": 0,
                "page_count": None,
            }
        docs[did]["chunk_count"] += 1

    return list(docs.values())
