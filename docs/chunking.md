# Hybrid Chunking Strategies

Chunking is the process of splitting large documents into smaller pieces (chunks) so they fit within an LLM's context window and improve embedding similarity search.

## Recursive Size-Based Chunking

The most common method is the recursive character text splitter. It splits text based on a size limit (e.g., 1000 characters) and a set of separators (like double newlines, single newlines, spaces).

**Formula for overlap:**
If chunk size is $S$ and overlap is $O$, the effective step size is $S - O$.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
```

## Structural Chunking

For technical documentation, structural chunking is superior. It splits documents based on semantic boundaries rather than arbitrary character counts.

A structural chunker will:
1. Split at `<h1>` or `<h2>` tags.
2. Ensure that Python classes and functions are kept intact.
3. Merge small consecutive sections (like an `<h3>` with only one sentence) into the parent chunk.

## The Hybrid Approach

A hybrid approach combines both methods:
1. Try to chunk **structurally** first.
2. If a structural block (like a massive API response JSON) exceeds the maximum chunk size, fall back to **recursive** chunking strictly *within* that block.

This ensures you never break the overall document flow, but you still respect the vector store's token limits.
