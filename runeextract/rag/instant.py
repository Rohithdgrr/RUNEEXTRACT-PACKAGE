"""
One-liner RAG pipeline factory.

``instant_rag()`` wraps ``AutoRAG`` with smarter defaults, domain templates,
and optional caching — getting from document to answer in a single call.
"""

import logging
from typing import Any, Dict, Optional, Union

from runeextract.rag.auto_pipeline import AutoRAG, auto_rag as _auto_rag
from runeextract.rag.embedding_selector import resolve_embedding
from runeextract.rag.templates import DomainConfig, DomainTemplates

logger = logging.getLogger(__name__)


def instant_rag(
    source: Union[str, list],
    model: str = "openai:gpt-4o-mini",
    domain: Optional[str] = None,
    store: str = "faiss",
    cache: bool = True,
    **kwargs: Any,
) -> AutoRAG:
    """Create a zero-config RAG pipeline and ingest documents immediately.

    Usage::

        rag = instant_rag("report.pdf", domain="financial")
        result = rag.query("What is the net income?")
        print(result.answer)

    Args:
        source: File path, directory, URL, or list of paths.
        model: LLM spec (``provider:model_name``).
        domain: Optional domain preset (``"financial"``, ``"legal"``,
            ``"medical"``, ``"academic"``). Sets chunking, size, overlap,
            reranker, and prompt defaults.
        store: Vector store (``"faiss"`` or ``"chromadb"``). Defaults to
            FAISS for lighter setup.
        cache: Enable multi-level caching.
        **kwargs: Additional arguments passed to ``AutoRAG``. Overrides
            any domain-derived defaults.

    Returns:
        An ``AutoRAG`` instance with documents already ingested.
    """
    config = DomainTemplates.get(domain) if domain else DomainConfig()

    embedding = resolve_embedding(
        kwargs.pop("embedding", config.embedding if domain else "balanced")
    )

    rag_kwargs: Dict[str, Any] = dict(
        embedding=embedding,
        vector_store=store,
        chunking=kwargs.pop("chunking", config.chunking),
        chunk_size=kwargs.pop("chunk_size", config.chunk_size),
        chunk_overlap=kwargs.pop("chunk_overlap", config.chunk_overlap),
        reranker=kwargs.pop("reranker", config.reranker),
        llm=model,
    )

    if cache:
        rag_kwargs.setdefault("ai_processor", None)

    rag_kwargs.update(kwargs)

    rag = _auto_rag(source, **rag_kwargs)

    if cache and config.system_prompt:
        try:
            if hasattr(rag, "_ai") and rag._ai is not None:
                rag._ai._system_prompt = config.system_prompt
        except Exception:
            logger.debug("Could not set system prompt from domain template")

    return rag
