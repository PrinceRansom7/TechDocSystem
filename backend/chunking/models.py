"""
TechChunk data model — the atomic unit of the vector store.

Field ontology is adapted from the regulatory Chunk model in Vec+GraphRAG
but repurposed for technical documentation:
  - 'regulator / jurisdiction / act_name' → 'library / version / module_path'
  - 'semantic_role' → technical roles (function_def, api_reference, code_example, …)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional


# Allowed semantic roles for technical documents
TECH_SEMANTIC_ROLES = frozenset({
    "overview",
    "installation",
    "usage",
    "api_reference",
    "code_example",
    "configuration",
    "parameter",
    "return_value",
    "exception",
    "note",
    "formula",
    "class_def",
    "function_def",
    "other",
})


@dataclass
class TechChunk:
    """
    A single chunk with text and rich technical metadata for the vector store.
    """

    # ── Identity ─────────────────────────────────────────────────────────────
    chunk_id: str
    document_id: str
    chunk_index: int
    total_chunks: int

    # ── Content ───────────────────────────────────────────────────────────────
    text: str

    # ── Document-level metadata ───────────────────────────────────────────────
    source: str                               # filename or URL
    source_type: str = "text"                 # pdf | markdown | text | html
    title: Optional[str] = None              # document title
    library: Optional[str] = None            # e.g. "LangChain", "FastAPI"
    version: Optional[str] = None            # e.g. "0.2.0"
    module_path: Optional[str] = None        # e.g. "langchain.vectorstores.chroma"
    language: str = "en"

    # ── Structural metadata ───────────────────────────────────────────────────
    section_label: Optional[str] = None      # heading / section title
    parent_path: Optional[str] = None        # breadcrumb path
    page_start: Optional[int] = None
    page_end: Optional[int] = None

    # ── Technical ontology ────────────────────────────────────────────────────
    semantic_role: str = "other"             # from TECH_SEMANTIC_ROLES
    has_code: bool = False                   # chunk contains a code snippet
    has_formula: bool = False                # chunk contains math/formula
    code_language: Optional[str] = None      # e.g. "python", "bash", "json"

    # ── Extra ─────────────────────────────────────────────────────────────────
    extra: dict[str, Any] = field(default_factory=dict)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a flat dict suitable for ChromaDB metadata."""
        d: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "text": self.text,
            "source": self.source,
            "source_type": self.source_type,
            "semantic_role": self.semantic_role,
            "has_code": self.has_code,
            "has_formula": self.has_formula,
            "language": self.language,
        }
        for key in ("title", "library", "version", "module_path",
                    "section_label", "parent_path",
                    "page_start", "page_end", "code_language"):
            v = getattr(self, key)
            if v is not None:
                d[key] = v
        if self.extra:
            d["extra"] = str(self.extra)
        return d

    @staticmethod
    def make_id(document_id: str, index: int) -> str:
        raw = f"{document_id}_{index}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    @staticmethod
    def document_id_from_source(source: str) -> str:
        return hashlib.sha256(source.encode("utf-8")).hexdigest()[:24]
