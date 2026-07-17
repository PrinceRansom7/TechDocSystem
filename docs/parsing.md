# Parsing Technical Documents

Document parsing is the process of converting raw unstructured files (PDFs, HTML, Markdown) into structured text and metadata. In RAG pipelines, parsing is often the most critical step for ensuring high data quality.

## Structural Boundaries

When parsing technical documents, it is important to preserve semantic boundaries. Unlike legal or regulatory documents that use strict section numbering (e.g., Article 1.1), technical documents rely on:

1. **Markdown Headings** (`#`, `##`, `###`)
2. **Code Fences** (```` ```python ````)
3. **API Endpoints** (`GET /users/{id}`)
4. **Formulas** (`$$ E = mc^2 $$`)

## Preserving Code Blocks

One of the biggest mistakes in technical parsing is treating code blocks as normal text. A parser should detect where a code block begins and ends so that chunking algorithms do not slice a function in half.

For example, given the following markdown:

```javascript
function greet(name) {
    console.log("Hello, " + name);
}
```

A good parser emits a `TechStructElement` indicating `has_code=True` and `code_language="javascript"`.

## Table Parsing

Tables in PDFs are notoriously difficult to parse because PDFs do not have a concept of rows and columns—only characters placed at specific XY coordinates. Tools like PyMuPDF or Docling use heuristic bounding boxes and reading orders to reconstruct tables into Markdown or CSV format before they are embedded into the vector store.
