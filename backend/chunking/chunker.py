"""
Technical document chunker.

Strategy hierarchy (mirrors Vec+GraphRAG structural/recursive/hybrid approach):

  hybrid   → try structural (heading-based); if <3 headings detected, fall back to recursive.
  structural → split at heading/function/class boundaries; merge small sections; split oversized ones.
  recursive  → size-based splitting with overlap, breaking at paragraph/line/sentence/word.

Special handling for technical content:
  - Fenced code blocks are NEVER split mid-block.
  - Formula blocks ($$ ... $$) are kept intact.
  - Code snippets are labelled with has_code=True and code_language detected.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from backend.parsing.extractor import ExtractedDocument
from backend.parsing.structure import (
    TechStructElement,
    build_parent_path,
    detect_tech_structure,
)
from backend.chunking.models import TechChunk, TECH_SEMANTIC_ROLES

logger = logging.getLogger(__name__)

# Recursive splitter separators — prefer paragraph → newline → sentence → word
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " "]

# Detect code fence language: ```python, ```bash, etc.
_CODE_LANG_RE = re.compile(r"^```(\w+)", re.MULTILINE)
_FORMULA_RE = re.compile(r"\$\$.+?\$\$", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]+`")


# ── Internal: recursive size-based split ──────────────────────────────────────

def _recursive_split(
    text: str,
    max_chars: int,
    overlap_chars: int,
    separators: list[str],
) -> list[tuple[int, int]]:
    """Return (start, end) char ranges, honouring code block boundaries."""
    if not text or max_chars <= 0:
        return []
    if len(text) <= max_chars:
        return [(0, len(text))]

    # Pre-compute code block spans so we never split inside them
    code_spans: list[tuple[int, int]] = [
        (m.start(), m.end()) for m in re.finditer(r"(```|~~~)[\s\S]*?\1", text)
    ]

    def is_code(pos: int) -> bool:
        return any(s <= pos < e for s, e in code_spans)

    ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # Never cut inside a code block
        for cs, ce in code_spans:
            if cs < end < ce:
                end = ce  # extend to end of code block
                break
        if end < len(text):
            chunk = text[start:end]
            best_sep = -1
            for sep in separators:
                pos = chunk.rfind(sep)
                if pos > max_chars // 2 and not is_code(start + pos):
                    best_sep = pos
                    break
                if pos > best_sep and not is_code(start + pos):
                    best_sep = pos
            if best_sep > 0:
                end = start + best_sep + 1
        ranges.append((start, end))
        start = max(start + 1, end - overlap_chars)
        if start >= len(text):
            break
    return ranges


# ── Internal: code & formula detection helpers ─────────────────────────────────

def _chunk_has_code(text: str) -> bool:
    return bool(re.search(r"```|~~~|^\s{4}", text, re.MULTILINE))


def _chunk_has_formula(text: str) -> bool:
    return bool(_FORMULA_RE.search(text))


def _detect_code_language(text: str) -> Optional[str]:
    m = _CODE_LANG_RE.search(text)
    return m.group(1).lower() if m else None


def _validate_role(role: Optional[str]) -> str:
    if role and role in TECH_SEMANTIC_ROLES:
        return role
    return "other"


# ── Recursive chunker ─────────────────────────────────────────────────────────

def chunk_recursive(
    extracted: ExtractedDocument,
    document_id: str,
    doc_meta: dict,
    max_chunk_chars: int,
    overlap_chars: int = 150,
    separators: Optional[list[str]] = None,
) -> list[TechChunk]:
    """Chunk by recursive size-based splitting with overlap."""
    if separators is None:
        separators = DEFAULT_SEPARATORS
    text = extracted.full_text.strip()
    if not text:
        return []

    ranges = _recursive_split(text, max_chunk_chars, overlap_chars, separators)
    chunks: list[TechChunk] = []
    for i, (s, e) in enumerate(ranges):
        chunk_text = text[s:e].strip()
        if not chunk_text:
            continue
        page = extracted.get_page_at_offset(s)
        cid = TechChunk.make_id(document_id, i)
        chunks.append(
            TechChunk(
                chunk_id=cid,
                document_id=document_id,
                chunk_index=len(chunks),
                total_chunks=len(ranges),
                text=chunk_text,
                source=doc_meta.get("source", ""),
                source_type=extracted.source_type,
                title=doc_meta.get("title"),
                library=doc_meta.get("library"),
                version=doc_meta.get("version"),
                section_label=None,
                parent_path=None,
                page_start=page,
                page_end=page,
                semantic_role="other",
                has_code=_chunk_has_code(chunk_text),
                has_formula=_chunk_has_formula(chunk_text),
                code_language=_detect_code_language(chunk_text),
            )
        )
    for c in chunks:
        c.total_chunks = len(chunks)
    return chunks


# ── Structural chunker ────────────────────────────────────────────────────────

def chunk_structural(
    extracted: ExtractedDocument,
    elements: list[TechStructElement],
    document_id: str,
    doc_meta: dict,
    max_chunk_chars: int,
    min_chunk_chars: int = 200,
    overlap_chars: int = 150,
    separators: Optional[list[str]] = None,
) -> list[TechChunk]:
    """
    Chunk at structural (heading / function / class) boundaries.
    Small adjacent sections are merged; large sections are recursively split.
    Code blocks within a section are never split.
    """
    if separators is None:
        separators = DEFAULT_SEPARATORS
    text = extracted.full_text
    chunks: list[TechChunk] = []
    chunk_index = 0

    # Prefer heading-level elements as primary boundaries
    heading_types = {"heading_h1", "heading_h2", "heading_h3", "heading_lower", "section_numbered"}
    primary = [e for e in elements if e.type in heading_types]

    if not primary:
        # No headings → recursive fallback
        return chunk_recursive(extracted, document_id, doc_meta, max_chunk_chars, overlap_chars, separators)

    i = 0
    while i < len(primary):
        el = primary[i]
        block_text = text[el.start_offset: el.end_offset].strip()
        parent_path = build_parent_path(primary, i)

        if len(block_text) <= max_chunk_chars:
            # Merge small adjacent sections
            merged = block_text
            j = i + 1
            end_el = el
            while j < len(primary) and len(merged) < min_chunk_chars:
                nxt = primary[j]
                nxt_text = text[nxt.start_offset: nxt.end_offset].strip()
                if len(merged) + len(nxt_text) + 2 <= max_chunk_chars:
                    merged = merged + "\n\n" + nxt_text
                    end_el = nxt
                    j += 1
                else:
                    break
            chunk_text = merged.strip()
            if chunk_text:
                cid = TechChunk.make_id(document_id, chunk_index)
                chunks.append(TechChunk(
                    chunk_id=cid,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    total_chunks=0,
                    text=chunk_text,
                    source=doc_meta.get("source", ""),
                    source_type=extracted.source_type,
                    title=doc_meta.get("title"),
                    library=doc_meta.get("library"),
                    version=doc_meta.get("version"),
                    section_label=el.label,
                    parent_path=parent_path or None,
                    page_start=el.page,
                    page_end=end_el.page,
                    semantic_role=_validate_role(el.semantic_role),
                    has_code=_chunk_has_code(chunk_text),
                    has_formula=_chunk_has_formula(chunk_text),
                    code_language=_detect_code_language(chunk_text),
                ))
                chunk_index += 1
            i = j
        else:
            # Oversized section — recursive sub-split, preserving code blocks
            sub_ranges = _recursive_split(block_text, max_chunk_chars, overlap_chars, separators)
            for s, e in sub_ranges:
                sub_text = block_text[s:e].strip()
                if not sub_text:
                    continue
                cid = TechChunk.make_id(document_id, chunk_index)
                chunks.append(TechChunk(
                    chunk_id=cid,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    total_chunks=0,
                    text=sub_text,
                    source=doc_meta.get("source", ""),
                    source_type=extracted.source_type,
                    title=doc_meta.get("title"),
                    library=doc_meta.get("library"),
                    version=doc_meta.get("version"),
                    section_label=el.label,
                    parent_path=parent_path or None,
                    page_start=el.page,
                    page_end=el.page,
                    semantic_role=_validate_role(el.semantic_role),
                    has_code=_chunk_has_code(sub_text),
                    has_formula=_chunk_has_formula(sub_text),
                    code_language=_detect_code_language(sub_text),
                ))
                chunk_index += 1
            i += 1

    for c in chunks:
        c.total_chunks = len(chunks)
    return chunks


# ── Hybrid dispatcher ─────────────────────────────────────────────────────────

def chunk_document(
    extracted: ExtractedDocument,
    document_id: str,
    doc_meta: dict,
    strategy: str = "hybrid",
    max_chunk_chars: int = 1500,
    min_chunk_chars: int = 200,
    overlap_chars: int = 150,
    separators: Optional[list[str]] = None,
) -> list[TechChunk]:
    """
    Chunk a technical document using the selected strategy.

    strategy:
      'recursive'  → pure size-based
      'structural' → heading/boundary-based
      'hybrid'     → structural if ≥3 headings detected, else recursive
    """
    if strategy == "recursive":
        return chunk_recursive(extracted, document_id, doc_meta, max_chunk_chars, overlap_chars, separators)

    elements = detect_tech_structure(extracted)

    if strategy == "structural":
        return chunk_structural(
            extracted, elements, document_id, doc_meta,
            max_chunk_chars, min_chunk_chars, overlap_chars, separators
        )

    # hybrid: count primary heading elements
    heading_types = {"heading_h1", "heading_h2", "heading_h3"}
    heading_count = sum(1 for e in elements if e.type in heading_types)
    if heading_count >= 3:
        logger.debug("Hybrid: %d headings found → structural chunking", heading_count)
        return chunk_structural(
            extracted, elements, document_id, doc_meta,
            max_chunk_chars, min_chunk_chars, overlap_chars, separators
        )

    logger.debug("Hybrid: only %d headings → recursive chunking", heading_count)
    return chunk_recursive(extracted, document_id, doc_meta, max_chunk_chars, overlap_chars, separators)
