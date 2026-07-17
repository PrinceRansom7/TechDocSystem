"""
Technical document structure detection.

Detects structural boundaries in technical documents:
  - H1–H6 headings (Markdown-style)
  - Numbered sections (1., 1.1, etc.)
  - Code block fences (``` / ~~~)
  - Class / function definitions
  - API endpoint patterns (GET /path, POST /path)
  - Formula / math blocks ($$ ... $$)
  - Note / warning / tip admonitions
  - Enumerated lists and parameter tables
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from backend.parsing.extractor import ExtractedDocument


@dataclass
class TechStructElement:
    """A detected structural boundary in a technical document."""

    # Type tags for technical docs (replaces legal part/chapter/section)
    type: str
    """
    One of:
      heading_h1 | heading_h2 | heading_h3 | heading_lower
      section_numbered | code_block | function_def | class_def
      api_endpoint | formula_block | admonition | list_block
    """
    label: str            # e.g. "Installation", "def connect()", "POST /query"
    title: Optional[str] = None
    start_offset: int = 0
    end_offset: int = 0
    page: Optional[int] = None
    semantic_role: str = "other"
    """
    Technical semantic roles (replaces legal obligation/penalty/proviso):
      overview | installation | usage | api_reference | code_example
      configuration | parameter | return_value | exception | note
      formula | class_def | function_def | other
    """


# ── Regex patterns for technical structure ────────────────────────────────────

TECH_PATTERNS: dict[str, list[re.Pattern]] = {
    "heading_h1": [
        re.compile(r"^#\s+(.+)$", re.MULTILINE),
    ],
    "heading_h2": [
        re.compile(r"^##\s+(.+)$", re.MULTILINE),
    ],
    "heading_h3": [
        re.compile(r"^###\s+(.+)$", re.MULTILINE),
    ],
    "heading_lower": [
        re.compile(r"^#{4,6}\s+(.+)$", re.MULTILINE),
    ],
    "section_numbered": [
        re.compile(r"^(\d+(?:\.\d+)*)\s+([A-Z][^\n]+)$", re.MULTILINE),
    ],
    "function_def": [
        re.compile(r"^(?:async\s+)?def\s+(\w+)\s*\(", re.MULTILINE),
        re.compile(r"^(?:public|private|protected|static)?\s*\w+\s+(\w+)\s*\([^)]*\)\s*\{", re.MULTILINE),
    ],
    "class_def": [
        re.compile(r"^class\s+(\w+)[\s:(]", re.MULTILINE),
    ],
    "api_endpoint": [
        re.compile(r"\b(GET|POST|PUT|DELETE|PATCH|HEAD)\s+(\/[\w\-/{}:?=&]+)", re.MULTILINE),
    ],
    "formula_block": [
        re.compile(r"\$\$.+?\$\$", re.DOTALL),
        re.compile(r"^\\\[.+?\\\]$", re.MULTILINE | re.DOTALL),
    ],
    "admonition": [
        re.compile(r"^(?:Note|Warning|Tip|Important|Caution|Danger|Hint)\s*[:：]", re.MULTILINE | re.IGNORECASE),
        re.compile(r"^>\s*\[!(NOTE|WARNING|TIP|IMPORTANT|CAUTION)\]", re.MULTILINE | re.IGNORECASE),
    ],
}

# Maps heading text keywords → semantic_role
_ROLE_KEYWORDS: dict[str, str] = {
    "overview": "overview",
    "introduction": "overview",
    "getting started": "installation",
    "installation": "installation",
    "install": "installation",
    "quick start": "installation",
    "setup": "installation",
    "usage": "usage",
    "how to": "usage",
    "example": "code_example",
    "tutorial": "code_example",
    "api": "api_reference",
    "endpoint": "api_reference",
    "reference": "api_reference",
    "configuration": "configuration",
    "config": "configuration",
    "parameter": "parameter",
    "argument": "parameter",
    "return": "return_value",
    "output": "return_value",
    "exception": "exception",
    "error": "exception",
    "note": "note",
    "warning": "note",
    "tip": "note",
    "formula": "formula",
    "equation": "formula",
    "math": "formula",
    "class": "class_def",
    "function": "function_def",
    "method": "function_def",
}


def _detect_semantic_role(type_: str, label: str) -> str:
    """Infer semantic role from element type and label text."""
    if type_ in ("function_def",):
        return "function_def"
    if type_ in ("class_def",):
        return "class_def"
    if type_ in ("api_endpoint",):
        return "api_reference"
    if type_ in ("formula_block",):
        return "formula"
    if type_ in ("admonition",):
        return "note"
    label_lower = label.lower()
    for keyword, role in _ROLE_KEYWORDS.items():
        if keyword in label_lower:
            return role
    return "other"


def detect_tech_structure(
    extracted: ExtractedDocument,
    patterns: Optional[dict[str, list[re.Pattern]]] = None,
) -> list[TechStructElement]:
    """
    Detect technical structural elements in extracted document text.

    Returns a list of TechStructElement ordered by start_offset, with
    end_offset set to the start of the next element (or end of text).
    Code blocks are intentionally NOT split — they are treated as atomic units.
    """
    if patterns is None:
        patterns = TECH_PATTERNS

    text = extracted.full_text
    elements: list[TechStructElement] = []
    seen: set[tuple[str, int]] = set()

    # First: identify fenced code block spans to avoid splitting them
    code_spans: list[tuple[int, int]] = []
    for m in re.finditer(r"(```|~~~)[\s\S]*?\1", text):
        code_spans.append((m.start(), m.end()))

    def is_inside_code(pos: int) -> bool:
        return any(start <= pos < end for start, end in code_spans)

    def add(typ: str, label: str, start: int, end: int, title: Optional[str] = None) -> None:
        if (typ, start) in seen:
            return
        if is_inside_code(start) and typ not in ("formula_block",):
            return  # Don't split inside code blocks
        seen.add((typ, start))
        page = extracted.get_page_at_offset(start) if extracted.pages else None
        role = _detect_semantic_role(typ, label)
        elements.append(
            TechStructElement(
                type=typ,
                label=(label or typ)[:200],
                title=title,
                start_offset=start,
                end_offset=end,
                page=page,
                semantic_role=role,
            )
        )

    for typ, pats in patterns.items():
        for pat in pats:
            for m in pat.finditer(text):
                if m.lastindex and m.lastindex >= 1:
                    label = m.group(1).strip()
                    title = m.group(2).strip() if m.lastindex >= 2 and m.group(2) else None
                else:
                    label = m.group(0)[:100]
                    title = None
                add(typ, label, m.start(), m.end(), title)

    elements.sort(key=lambda e: (e.start_offset, e.type))

    # Set end offsets: each element extends to the start of the next one
    for i, el in enumerate(elements):
        if i + 1 < len(elements):
            el.end_offset = elements[i + 1].start_offset
        else:
            el.end_offset = len(text)

    return elements


def build_parent_path(elements: list[TechStructElement], up_to_index: int) -> str:
    """Build a breadcrumb path like 'Installation > Quick Start > pip install'."""
    parts: list[str] = []
    heading_levels = ("heading_h1", "heading_h2", "heading_h3", "heading_lower", "section_numbered")
    for i in range(up_to_index + 1):
        e = elements[i]
        if e.type in heading_levels:
            level_depth = {
                "heading_h1": 1,
                "heading_h2": 2,
                "heading_h3": 3,
                "heading_lower": 4,
                "section_numbered": 2,
            }.get(e.type, 2)
            # Truncate stack at this depth
            parts = parts[: level_depth - 1] + [e.label]
    return " > ".join(parts) if parts else ""
