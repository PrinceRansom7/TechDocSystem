"""
Technical document text extraction.

Supports:
  - PDF files (PyMuPDF / fitz)
  - Markdown files (plain text, preserving fenced code blocks)
  - Plain text files
  - HTML pages fetched from URLs (html2text for clean markdown output)

All extractors return an ExtractedDocument with full_text, per-page boundaries,
and detected content type hints for downstream chunking.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import requests

logger = logging.getLogger(__name__)

PAGE_BREAK = "\f"


@dataclass
class PageContent:
    """Text content for a single page (or logical section for non-PDFs)."""
    page_number: int
    text: str


@dataclass
class ExtractedDocument:
    """
    Full extracted text with per-page/section boundaries.

    Attributes:
        full_text: Concatenated text of all pages/sections.
        pages: Individual page/section content.
        num_pages: Total number of pages/sections.
        source_type: 'pdf' | 'markdown' | 'text' | 'html'
        title: Inferred document title (first heading or filename).
        has_code: Whether code blocks were detected.
        has_formulas: Whether mathematical notation was detected.
    """
    full_text: str
    pages: list[PageContent]
    num_pages: int
    source_type: str = "text"
    title: Optional[str] = None
    has_code: bool = False
    has_formulas: bool = False

    def get_page_at_offset(self, offset: int) -> Optional[int]:
        """Return 1-based page number for a character offset into full_text."""
        if offset < 0 or not self.pages:
            return None
        cumul = 0
        for p in self.pages:
            end = cumul + len(p.text) + 1  # +1 for page break char
            if offset < end:
                return p.page_number
            cumul = end
        return self.pages[-1].page_number if self.pages else None


# ── PDF extraction ─────────────────────────────────────────────────────────────

def extract_pdf(path: Path, skip_pages: int = 0) -> Optional[ExtractedDocument]:
    """Extract text from a PDF with per-page boundaries using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("PyMuPDF required: pip install PyMuPDF")
        return None

    path = Path(path)
    if not path.exists():
        logger.warning("PDF not found: %s", path)
        return None

    try:
        doc = fitz.open(str(path))
    except Exception as e:
        logger.warning("Failed to open PDF %s: %s", path, e)
        return None

    pages: list[PageContent] = []
    parts: list[str] = []
    try:
        for i in range(len(doc)):
            if i < skip_pages:
                continue
            page = doc[i]
            text = page.get_text().strip()
            pages.append(PageContent(page_number=i + 1, text=text))
            if parts:
                parts.append(PAGE_BREAK)
            parts.append(text)
        doc.close()
    except Exception as e:
        logger.warning("Error reading PDF %s: %s", path, e)
        doc.close()
        return None

    full_text = "".join(parts)
    title = _infer_title_from_text(full_text) or path.stem
    has_code, has_formulas = _detect_content_hints(full_text)
    return ExtractedDocument(
        full_text=full_text,
        pages=pages,
        num_pages=len(pages),
        source_type="pdf",
        title=title,
        has_code=has_code,
        has_formulas=has_formulas,
    )


# ── Markdown extraction ────────────────────────────────────────────────────────

def extract_markdown(path: Path) -> Optional[ExtractedDocument]:
    """Load a Markdown file, preserving fenced code blocks as-is."""
    path = Path(path)
    if not path.exists():
        logger.warning("Markdown file not found: %s", path)
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return None

    title = _extract_md_title(text) or path.stem
    has_code, has_formulas = _detect_content_hints(text)
    # Treat each H1/H2 heading as a logical "page"
    pages = _md_to_pages(text)
    return ExtractedDocument(
        full_text=text,
        pages=pages,
        num_pages=len(pages),
        source_type="markdown",
        title=title,
        has_code=has_code,
        has_formulas=has_formulas,
    )


# ── Plain text extraction ──────────────────────────────────────────────────────

def extract_text_file(path: Path) -> Optional[ExtractedDocument]:
    """Load a plain .txt file as a single-page document."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return None

    has_code, has_formulas = _detect_content_hints(text)
    pages = [PageContent(page_number=1, text=text)]
    return ExtractedDocument(
        full_text=text,
        pages=pages,
        num_pages=1,
        source_type="text",
        title=path.stem,
        has_code=has_code,
        has_formulas=has_formulas,
    )


# ── URL / HTML extraction ──────────────────────────────────────────────────────

def extract_url(url: str, timeout: int = 15) -> Optional[ExtractedDocument]:
    """Fetch a URL and convert HTML to clean Markdown-like text."""
    try:
        import html2text
    except ImportError:
        logger.error("html2text required: pip install html2text")
        return None

    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "TechDocRAG/1.0"})
        resp.raise_for_status()
    except Exception as e:
        logger.warning("Failed to fetch %s: %s", url, e)
        return None

    content_type = resp.headers.get("content-type", "")
    if "text/html" in content_type or "text/plain" in content_type:
        raw = resp.text
    else:
        logger.warning("Unsupported content-type %s for URL %s", content_type, url)
        return None

    # Convert HTML → clean text (preserving code blocks)
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.body_width = 0  # No wrapping
    h.protect_links = True
    h.wrap_links = False
    text = h.handle(raw)

    title = _extract_md_title(text) or url.split("/")[-1] or url
    has_code, has_formulas = _detect_content_hints(text)
    pages = _md_to_pages(text)
    return ExtractedDocument(
        full_text=text,
        pages=pages,
        num_pages=max(len(pages), 1),
        source_type="html",
        title=title,
        has_code=has_code,
        has_formulas=has_formulas,
    )


# ── Auto-dispatch ──────────────────────────────────────────────────────────────

def extract_document(source: str | Path, is_url: bool = False) -> Optional[ExtractedDocument]:
    """
    Auto-dispatch to the correct extractor based on source type.

    Args:
        source: A file path (str or Path) or a URL string.
        is_url: Force URL mode even if source looks like a path.
    """
    if is_url or (isinstance(source, str) and source.startswith(("http://", "https://"))):
        return extract_url(str(source))

    p = Path(source)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(p)
    elif suffix in (".md", ".mdx"):
        return extract_markdown(p)
    elif suffix in (".txt", ".rst", ".log"):
        return extract_text_file(p)
    else:
        # Attempt as plain text
        logger.info("Unknown extension %s, trying plain text extractor", suffix)
        return extract_text_file(p)


# ── Internal helpers ───────────────────────────────────────────────────────────

_CODE_PATTERNS = [
    re.compile(r"```[\s\S]*?```"),            # fenced code blocks
    re.compile(r"`[^`]+`"),                   # inline code
    re.compile(r"    [^\n]+"),                # indented code (4-space)
    re.compile(r"def\s+\w+\s*\("),           # Python function
    re.compile(r"class\s+\w+[\s:(]"),        # Python/Java class
    re.compile(r"import\s+\w+|from\s+\w+\s+import"),  # imports
    re.compile(r"<\w+>.*?</\w+>", re.DOTALL), # XML/HTML tags
]

_FORMULA_PATTERNS = [
    re.compile(r"\$\$.+?\$\$", re.DOTALL),   # LaTeX display math
    re.compile(r"\$.+?\$"),                   # LaTeX inline math
    re.compile(r"\\[a-zA-Z]+\{"),            # LaTeX commands
    re.compile(r"O\([nN^]\)"),               # Big-O notation
    re.compile(r"\b[A-Z][a-z]*_\{?\d"),     # subscript notation
]


def _detect_content_hints(text: str) -> tuple[bool, bool]:
    has_code = any(p.search(text) for p in _CODE_PATTERNS)
    has_formulas = any(p.search(text) for p in _FORMULA_PATTERNS)
    return has_code, has_formulas


def _infer_title_from_text(text: str) -> Optional[str]:
    """Try to pick the first meaningful line as title."""
    for line in text.splitlines():
        line = line.strip()
        if line and len(line) > 3 and not line.startswith(("http", "#")):
            return line[:120]
    return None


def _extract_md_title(text: str) -> Optional[str]:
    """Extract the first H1 heading from Markdown text."""
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else None


def _md_to_pages(text: str) -> list[PageContent]:
    """
    Split Markdown text into logical pages at H1/H2 headings.
    Each heading starts a new "page". Code blocks are never split.
    """
    # State machine: skip heading detection inside fenced code blocks
    lines = text.splitlines(keepends=True)
    sections: list[list[str]] = [[]]
    in_fence = False
    fence_char = ""

    for line in lines:
        stripped = line.strip()
        # Detect fenced code block boundaries
        if stripped.startswith("```") or stripped.startswith("~~~"):
            ch = stripped[:3]
            if not in_fence:
                in_fence = True
                fence_char = ch
            elif ch == fence_char:
                in_fence = False
            sections[-1].append(line)
            continue

        if not in_fence and re.match(r"^#{1,2}\s+", line):
            if sections[-1]:  # Start new section only if current is non-empty
                sections.append([])

        sections[-1].append(line)

    pages = []
    for i, sec_lines in enumerate(sections):
        text_section = "".join(sec_lines).strip()
        if text_section:
            pages.append(PageContent(page_number=i + 1, text=text_section))

    return pages if pages else [PageContent(page_number=1, text=text)]
