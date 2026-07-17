"""
LangGraph StateGraph — self-corrective RAG workflow.

Nodes:
  1. query_analysis   → classify + rewrite/expand query for retrieval
  2. retrieval        → fetch top-k chunks from ChromaDB
  3. document_grading → LLM scores each chunk; filters irrelevants
  4. web_search       → optional Tavily / DuckDuckGo fallback
  5. generation       → synthesise answer with citations

Conditional routing:
  After grading:
    - relevant docs found              → generation
    - no relevant, retries < max       → query_analysis (rewrite loop)
    - no relevant, retries >= max, web → web_search → generation
    - no relevant, retries >= max, no web → fallback "I don't know"
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal, Optional, TypedDict
import operator

from langgraph.graph import END, StateGraph
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from backend.config import Settings

logger = logging.getLogger(__name__)


# ── State schema ──────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    # Input
    original_question: str
    response_mode: str            # "concise" | "detailed"
    enable_web_search: bool
    session_history: list[dict]   # [{role, content}] for conversation memory
    force_rewrite: bool           # human-in-the-loop override
    expand_query: bool            # user-requested synonym expansion

    # Working state
    query: str                    # current (possibly rewritten) query
    retries: int
    retrieved_docs: list[dict]    # [{text, source, metadata}]
    graded_docs: list[dict]       # filtered subset
    discarded_docs: list[dict]    # docs filtered out by LLM
    web_results: list[dict]       # [{content, source}]

    # Output
    answer: str
    query_type: str               # CONCEPTUAL | HOW_TO | TROUBLESHOOTING | API_REFERENCE | COMPARATIVE | DEFINITIONAL | DEBUGGING | GENERAL_CHAT
    fallback_used: bool
    hallucination_status: str     # "pass" | "fail" | "unchecked"
    hallucination_retries: int


# ── LLM factory (Groq) ────────────────────────────────────────────────────────

def _make_llm(settings: Settings, temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=temperature,
    )


# ── Helper: embed query ───────────────────────────────────────────────────────

def _embed_query(query: str, settings: Settings) -> list[float]:
    """Embed a single query using sentence-transformers (reuses the cached model)."""
    from backend.ingest import _get_st_model
    model = _get_st_model(settings.embedding_model)
    emb = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return emb[0].tolist()


# ── Helper: retrieve from ChromaDB ────────────────────────────────────────────

def _retrieve_from_chroma(query: str, settings: Settings) -> list[dict]:
    import chromadb
    client = chromadb.PersistentClient(path=str(settings.chroma_persist_path))
    col = client.get_or_create_collection(name=settings.chroma_collection)

    if col.count() == 0:
        return []

    emb = _embed_query(query, settings)
    results = col.query(
        query_embeddings=[emb],
        n_results=min(settings.top_k, col.count()),
        include=["documents", "metadatas", "distances"],
    )
    docs: list[dict] = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        docs.append({
            "text": text,
            "source": meta.get("source", "unknown"),
            "section_label": meta.get("section_label"),
            "parent_path": meta.get("parent_path"),
            "semantic_role": meta.get("semantic_role", "other"),
            "document_id": meta.get("document_id"),
            "page_start": meta.get("page_start"),
            "page_end": meta.get("page_end"),
            "has_code": meta.get("has_code", False),
            "has_formula": meta.get("has_formula", False),
            "distance": dist,
        })
    return docs


# ── Node 1: Query Analysis ────────────────────────────────────────────────────

def node_query_analysis(state: RAGState, settings: Settings) -> dict:
    """
    Classify the query and rewrite/expand it for better retrieval.
    On first run: classify + rewrite.
    On retry: rewrite with hint to diversify.
    """
    llm = _make_llm(settings)
    original = state["original_question"]
    retries = state.get("retries", 0)

    # Step 1: Fine-grained classification
    classify_prompt = f"""You are a query classifier for a technical documentation assistant.

Classify the query into EXACTLY ONE of these categories:
- GENERAL_CHAT: Greetings, small talk, off-topic, or non-technical questions.
- CONCEPTUAL: Questions asking what something is, how it works, or its purpose (e.g. "What is X?").
- HOW_TO: Step-by-step procedural questions (e.g. "How do I install/configure/use X?").
- TROUBLESHOOTING: Error diagnosis or debugging questions (e.g. "Why is X failing?", "Fix this error").
- API_REFERENCE: Questions about specific functions, classes, parameters, or return values.
- COMPARATIVE: Questions comparing two or more things (e.g. "X vs Y", "difference between").
- DEFINITIONAL: Questions asking for definitions, terminology, or acronym expansions.
- DEBUGGING: Questions involving code that doesn't work as expected, stack traces, or runtime issues.

Query: {original}

Respond with ONLY the category label (e.g. HOW_TO)."""

    raw_classification = llm.invoke(classify_prompt).content.strip().upper()
    valid_types = {"GENERAL_CHAT", "CONCEPTUAL", "HOW_TO", "TROUBLESHOOTING", "API_REFERENCE", "COMPARATIVE", "DEFINITIONAL", "DEBUGGING"}
    query_type = raw_classification if raw_classification in valid_types else "CONCEPTUAL"

    if query_type == "GENERAL_CHAT":
        return {"query": original, "query_type": query_type}

    # Step 2: rewrite for better retrieval
    expand_hint = ""
    if state.get("expand_query"):
        expand_hint = "\n\nIMPORTANT: The user has requested QUERY EXPANSION. Aggressively add synonyms, related concepts, alternative phrasings, and clarify any ambiguity. Make the query as broad and semantically rich as possible while staying on topic."
    retry_hint = ""
    if state.get("force_rewrite"):
        retry_hint = "\n\nCRITICAL: The user has explicitly requested ONE MORE aggressive rewrite because previous attempts failed. Completely change the keywords, abstract the concept, or focus on a different synonym."
    elif retries > 0:
        retry_hint = f"\n\nPrevious retrieval attempt #{retries} returned no relevant results. Try different keywords, synonyms, or a more specific/general phrasing."

    rewrite_prompt = f"""You are an expert at expanding and rewriting technical questions to maximise semantic search recall.

Rewrite the following question to:
1. Add relevant technical synonyms (e.g., "function" -> "method, callable, def")
2. Include the likely library/framework name if implied
3. Mention related concepts (e.g., "how to connect" -> "connection setup configuration credentials")
4. Keep it concise (max 2 sentences){expand_hint}{retry_hint}

Original question: {original}

Return ONLY the rewritten question, nothing else."""

    rewritten = llm.invoke(rewrite_prompt).content.strip()
    logger.info("Query rewritten [retry=%d]: %s → %s", retries, original, rewritten)

    return {"query": rewritten, "query_type": query_type}


# ── Node 2: Retrieval ─────────────────────────────────────────────────────────

def node_retrieval(state: RAGState, settings: Settings) -> dict:
    """Retrieve top-k chunks from ChromaDB using the (possibly rewritten) query."""
    query = state.get("query", state["original_question"])
    docs = _retrieve_from_chroma(query, settings)
    logger.info("Retrieved %d chunks for query: %s", len(docs), query[:80])
    return {"retrieved_docs": docs}


# ── Node 3: Document Grading ──────────────────────────────────────────────────

def node_document_grading(state: RAGState, settings: Settings) -> dict:
    """
    Score each retrieved chunk as 'relevant' or 'irrelevant'.
    Filter and return only relevant ones.
    """
    llm = _make_llm(settings)
    question = state["original_question"]
    docs = state.get("retrieved_docs", [])

    if not docs:
        return {"graded_docs": []}

    graded: list[dict] = []
    discarded: list[dict] = []
    for doc in docs:
        grade_prompt = f"""You are a relevance grader for a technical documentation assistant.

Score whether the following document chunk is RELEVANT to the user's question.

Question: {question}

Document chunk:
\"\"\"
{doc['text'][:800]}
\"\"\"

Respond with ONLY one word: "relevant" or "irrelevant"."""

        verdict = llm.invoke(grade_prompt).content.strip().lower()
        if "relevant" in verdict and "irrelevant" not in verdict:
            graded.append(doc)
        else:
            logger.debug("Chunk from %s graded irrelevant", doc["source"])
            discarded.append(doc)

    logger.info("%d/%d chunks graded relevant", len(graded), len(docs))
    return {"graded_docs": graded, "discarded_docs": discarded}


# ── Node 4: Web Search (fallback) ────────────────────────────────────────────

def node_web_search(state: RAGState, settings: Settings) -> dict:
    """
    Web search fallback. Uses Tavily if TAVILY_API_KEY is set,
    otherwise falls back to DuckDuckGo (free, no key required).
    """
    query = state.get("query", state["original_question"])
    results: list[dict] = []

    if settings.tavily_api_key:
        try:
            from tavily import TavilyClient
            tc = TavilyClient(api_key=settings.tavily_api_key)
            resp = tc.search(query=query, max_results=3, search_depth="advanced")
            for r in resp.get("results", []):
                results.append({"content": r.get("content", ""), "source": r.get("url", "")})
            logger.info("Tavily returned %d results", len(results))
        except Exception as e:
            logger.warning("Tavily search failed: %s. Falling back to DuckDuckGo.", e)

    if not results:
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=3):
                    results.append({"content": r.get("body", ""), "source": r.get("href", "")})
            logger.info("DuckDuckGo returned %d results", len(results))
        except Exception as e:
            logger.warning("DuckDuckGo search also failed: %s", e)

    return {"web_results": results}


# ── Node 5: Generation ────────────────────────────────────────────────────────

def node_generation(state: RAGState, settings: Settings) -> dict:
    """
    Generate the final answer grounded in retrieved context, with citations.
    Respects response_mode: concise → direct answer; detailed → full explanation.
    """
    llm = _make_llm(settings, temperature=0.1)
    question = state["original_question"]
    mode = state.get("response_mode", "concise")
    query_type = state.get("query_type", "TECHNICAL_QUERY")
    history = state.get("session_history", [])

    # General chat — no RAG needed
    if query_type == "GENERAL_CHAT":
        conv = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history[-6:])
        prompt = f"{conv}\nUser: {question}" if conv else question
        answer = llm.invoke(prompt).content
        return {"answer": answer, "fallback_used": False}

    graded = state.get("graded_docs", [])
    web = state.get("web_results", [])

    # Fallback: no relevant docs and no web results
    if not graded and not web:
        return {
            "answer": (
                "⚠️ **I don't have enough information to answer this question.**\n\n"
                "The documents currently in the knowledge base do not contain relevant information "
                "for your query, and web search did not return useful results.\n\n"
                "**Suggestions:**\n"
                "- Try rephrasing your question with different keywords.\n"
                "- Upload or ingest the relevant documentation using the sidebar.\n"
                "- Enable **Web Search** for live results from the internet."
            ),
            "fallback_used": True,
        }

    # Build context from graded docs
    doc_context = ""
    citations: list[str] = []
    for i, doc in enumerate(graded):
        label = f"[Doc {i+1}]"
        source_line = f"{doc['source']}"
        if doc.get("section_label"):
            source_line += f" § {doc['section_label']}"
        doc_context += f"\n\n{label} ({source_line}):\n{doc['text']}"
        citations.append(f"{label} {source_line}")

    # Append web context if any
    web_context = ""
    if web:
        web_context = "\n\n--- Web Search Results ---"
        for j, w in enumerate(web):
            web_context += f"\n\n[Web {j+1}] ({w['source']}):\n{w['content'][:600]}"

    # Response style instruction
    if mode == "detailed":
        style = (
            "Provide a thorough, well-structured explanation. "
            "Include all relevant details, step-by-step breakdowns where appropriate, "
            "code examples from the context, and clearly cite your sources using [Doc N] notation."
        )
    else:
        style = (
            "Provide a short, direct answer. "
            "Cite the source document(s) using [Doc N] notation. "
            "If code is relevant, include only the essential snippet."
        )

    # Conversation history context
    conv_ctx = ""
    if history:
        conv_ctx = "\n\nConversation history (for follow-up context):\n"
        conv_ctx += "\n".join(f"{m['role'].capitalize()}: {m['content'][:200]}" for m in history[-4:])

    sys_prompt = (
        "You are a precise technical documentation assistant. "
        "Answer questions ONLY based on the provided document context. "
        "Never invent information. If context is insufficient, say so clearly."
    )

    user_prompt = f"""{style}

Document context:{doc_context}{web_context}{conv_ctx}

Question: {question}

Answer (with [Doc N] citations):"""

    messages = [SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)]
    answer = llm.invoke(messages).content

    return {"answer": answer, "fallback_used": False, "hallucination_status": "unchecked"}


# ── Node 6: Hallucination Check (Self-RAG ISSUP) ─────────────────────────

def node_hallucination_check(state: RAGState, settings: Settings) -> dict:
    """
    Inspired by Self-RAG ISSUP token: verify that every claim in the generated
    answer is actually supported by the retrieved context (graded_docs + web_results).
    Returns hallucination_status: 'pass' | 'fail'.
    """
    llm = _make_llm(settings, temperature=0.0)
    answer = state.get("answer", "")
    graded = state.get("graded_docs", [])
    web = state.get("web_results", [])

    if not graded and not web:
        # No context to check against — skip
        return {"hallucination_status": "pass"}

    context_parts = []
    for doc in graded:
        context_parts.append(f"[Doc] {doc['source']}:\n{doc['text'][:600]}")
    for w in web:
        context_parts.append(f"[Web] {w['source']}:\n{w['content'][:400]}")
    context_str = "\n\n".join(context_parts)

    check_prompt = f"""You are a strict factual grounding evaluator (Self-RAG ISSUP).

Your task: determine whether the GENERATED ANSWER is fully supported by the PROVIDED CONTEXT.

Rules:
- Answer PASS if every factual claim in the answer can be traced to the context.
- Answer FAIL if the answer contains ANY claim, detail, or fact not present in the context (hallucination).
- Ignore phrasing and style differences; focus only on factual accuracy.

PROVIDED CONTEXT:
{context_str}

GENERATED ANSWER:
{answer}

Respond with ONLY one word: PASS or FAIL."""

    verdict = llm.invoke(check_prompt).content.strip().upper()
    status = "pass" if verdict == "PASS" else "fail"
    logger.info("Hallucination check: %s", status)
    return {"hallucination_status": status}


# ── Routing logic ─────────────────────────────────────────────────────────────

def route_after_grading(state: RAGState, settings: Settings) -> str:
    """
    Decide next node after document grading.

    Returns one of: "generation" | "query_analysis" | "web_search" | "fallback_generation"
    """
    graded = state.get("graded_docs", [])
    retries = state.get("retries", 0)
    max_retries = settings.max_retries
    enable_web = state.get("enable_web_search", False)
    query_type = state.get("query_type", "TECHNICAL_QUERY")

    # General chat always goes to generation
    if query_type == "GENERAL_CHAT":
        return "generation"

    if graded:
        logger.info("Routing → generation (%d relevant docs)", len(graded))
        return "generation"

    if retries < max_retries:
        logger.info("Routing → query_analysis (retry %d/%d)", retries + 1, max_retries)
        return "query_analysis"

    if enable_web:
        logger.info("Routing → web_search (retries exhausted)")
        return "web_search"

    logger.info("Routing → generation (fallback I-don't-know)")
    return "generation"  # generation node handles fallback message


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(settings: Settings) -> Any:
    """Compile and return the LangGraph StateGraph."""

    # Wrap each node to inject settings
    def _query_analysis(state):
        result = node_query_analysis(state, settings)
        return {**result, "retries": state.get("retries", 0) + (1 if state.get("retries", 0) > 0 else 0)}

    def _retrieval(state):
        return node_retrieval(state, settings)

    def _grading(state):
        return node_document_grading(state, settings)

    def _web_search(state):
        return node_web_search(state, settings)

    def _generation(state):
        return node_generation(state, settings)

    def _increment_retry(state):
        return {"retries": state.get("retries", 0) + 1}

    def _route(state) -> str:
        return route_after_grading(state, settings)

    def _hallucination_check(state):
        return node_hallucination_check(state, settings)

    def _route_after_hallucination(state) -> str:
        h_retries = state.get("hallucination_retries", 0)
        if state.get("hallucination_status") == "fail" and h_retries < 1:
            logger.info("Hallucination detected, regenerating (attempt %d)", h_retries + 1)
            return "generation"
        return "end"

    def _increment_hallucination_retry(state):
        return {"hallucination_retries": state.get("hallucination_retries", 0) + 1}

    builder = StateGraph(RAGState)

    builder.add_node("query_analysis", _query_analysis)
    builder.add_node("retrieval", _retrieval)
    builder.add_node("document_grading", _grading)
    builder.add_node("web_search", _web_search)
    builder.add_node("generation", _generation)
    builder.add_node("increment_retry", _increment_retry)
    builder.add_node("hallucination_check", _hallucination_check)
    builder.add_node("increment_hallucination_retry", _increment_hallucination_retry)

    builder.set_entry_point("query_analysis")

    builder.add_edge("query_analysis", "retrieval")
    builder.add_edge("retrieval", "document_grading")

    builder.add_conditional_edges(
        "document_grading",
        _route,
        {
            "generation": "generation",
            "query_analysis": "increment_retry",
            "web_search": "web_search",
        },
    )

    builder.add_edge("increment_retry", "query_analysis")
    builder.add_edge("web_search", "generation")
    builder.add_edge("generation", "hallucination_check")

    builder.add_conditional_edges(
        "hallucination_check",
        _route_after_hallucination,
        {
            "generation": "increment_hallucination_retry",
            "end": END,
        },
    )
    builder.add_edge("increment_hallucination_retry", "generation")

    return builder.compile()


# ── Public run function ───────────────────────────────────────────────────────

def run_rag_pipeline(
    question: str,
    settings: Settings,
    response_mode: str = "concise",
    enable_web_search: bool = False,
    session_history: Optional[list[dict]] = None,
) -> dict:
    """
    Entry point for the RAG pipeline.

    Returns:
        dict with keys: answer, query_type, sources, web_sources, retries_used, fallback_used
    """
    graph = build_graph(settings)
    initial_state: RAGState = {
        "original_question": question,
        "response_mode": response_mode,
        "enable_web_search": enable_web_search,
        "session_history": session_history or [],
        "force_rewrite": False,
        "expand_query": False,
        "query": question,
        "retries": 0,
        "retrieved_docs": [],
        "graded_docs": [],
        "discarded_docs": [],
        "web_results": [],
        "answer": "",
        "query_type": "CONCEPTUAL",
        "fallback_used": False,
        "hallucination_status": "unchecked",
        "hallucination_retries": 0,
    }

    final_state = graph.invoke(initial_state)

    # Build source list from graded docs
    sources = []
    for doc in final_state.get("graded_docs", []):
        sources.append({
            "document_id": doc.get("document_id", ""),
            "source": doc.get("source", ""),
            "section_label": doc.get("section_label"),
            "parent_path": doc.get("parent_path"),
            "semantic_role": doc.get("semantic_role"),
            "page_start": doc.get("page_start"),
            "page_end": doc.get("page_end"),
            "chunk_text": doc.get("text", ""),
        })

    discarded_sources = []
    for doc in final_state.get("discarded_docs", []):
        discarded_sources.append({
            "document_id": doc.get("document_id", ""),
            "source": doc.get("source", ""),
            "section_label": doc.get("section_label"),
            "parent_path": doc.get("parent_path"),
            "semantic_role": doc.get("semantic_role"),
            "page_start": doc.get("page_start"),
            "page_end": doc.get("page_end"),
            "chunk_text": doc.get("text", ""),
        })

    return {
        "answer": final_state.get("answer", ""),
        "query_type": final_state.get("query_type", "TECHNICAL_QUERY"),
        "sources": sources,
        "discarded_sources": discarded_sources,
        "web_sources": final_state.get("web_results", []),
        "hallucination_status": final_state.get("hallucination_status", "unchecked"),
        "retries_used": final_state.get("retries", 0),
        "fallback_used": final_state.get("fallback_used", False),
    }


def run_rag_pipeline_stream(
    question: str,
    settings: Settings,
    response_mode: str = "concise",
    enable_web_search: bool = False,
    session_history: Optional[list[dict]] = None,
    force_rewrite: bool = False,
    expand_query: bool = False,
):
    import json
    graph = build_graph(settings)
    initial_state: RAGState = {
        "original_question": question,
        "response_mode": response_mode,
        "enable_web_search": enable_web_search,
        "session_history": session_history or [],
        "force_rewrite": force_rewrite,
        "expand_query": expand_query,
        "query": question,
        "retries": 0,
        "retrieved_docs": [],
        "graded_docs": [],
        "discarded_docs": [],
        "web_results": [],
        "answer": "",
        "query_type": "CONCEPTUAL",
        "fallback_used": False,
        "hallucination_status": "unchecked",
        "hallucination_retries": 0,
    }

    final_state = initial_state.copy()
    for event in graph.stream(initial_state, stream_mode="updates"):
        node_name = list(event.keys())[0]
        node_update = event[node_name]
        final_state.update(node_update)

        safe_event = {
            "node": node_name,
            "update": node_update
        }
        yield f"data: {json.dumps(safe_event)}\n\n"

    sources = []
    for doc in final_state.get("graded_docs", []):
        sources.append({
            "document_id": doc.get("document_id", ""),
            "source": doc.get("source", ""),
            "section_label": doc.get("section_label"),
            "parent_path": doc.get("parent_path"),
            "semantic_role": doc.get("semantic_role"),
            "page_start": doc.get("page_start"),
            "page_end": doc.get("page_end"),
            "chunk_text": doc.get("text", ""),
        })

    discarded_sources = []
    for doc in final_state.get("discarded_docs", []):
        discarded_sources.append({
            "document_id": doc.get("document_id", ""),
            "source": doc.get("source", ""),
            "section_label": doc.get("section_label"),
            "parent_path": doc.get("parent_path"),
            "semantic_role": doc.get("semantic_role"),
            "page_start": doc.get("page_start"),
            "page_end": doc.get("page_end"),
            "chunk_text": doc.get("text", ""),
        })

    final_result = {
        "node": "final_result",
        "result": {
            "answer": final_state.get("answer", ""),
            "query_type": final_state.get("query_type", "CONCEPTUAL"),
            "sources": sources,
            "discarded_sources": discarded_sources,
            "web_sources": final_state.get("web_results", []),
            "hallucination_status": final_state.get("hallucination_status", "unchecked"),
            "retries_used": final_state.get("retries", 0),
            "fallback_used": final_state.get("fallback_used", False),
        }
    }
    yield f"data: {json.dumps(final_result)}\n\n"
