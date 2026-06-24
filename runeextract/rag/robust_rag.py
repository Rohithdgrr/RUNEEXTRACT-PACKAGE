"""
Graceful-degradation RAG pipeline with fallback strategies.

``RobustRAG`` wraps ``AutoRAG`` and provides fallback chains so that
a query still produces an answer even when the primary retriever or
LLM call fails.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from runeextract.processors.ai import AIProcessor
from runeextract.rag.auto_pipeline import AutoRAG
from runeextract.rag.types import RAGResult, ChunkWithScore

logger = logging.getLogger(__name__)


@dataclass
class FallbackStrategy:
    name: str
    enabled: bool = True
    max_retries: int = 1


_DEFAULT_STRATEGIES = [
    FallbackStrategy("primary_retriever"),
    FallbackStrategy("keyword_fallback"),
    FallbackStrategy("llm_only"),
]


class RobustRAG:
    """AutoRAG wrapper with automatic fallback on failure.

    Usage::

        base = AutoRAG()
        base.ingest("docs.pdf")
        rag = RobustRAG(base)
        result = rag.query("What is X?")
    """

    def __init__(self, pipeline: AutoRAG,
                 strategies: Optional[List[FallbackStrategy]] = None):
        self._pipeline = pipeline
        self._strategies = strategies or list(_DEFAULT_STRATEGIES)
        self._fallback_used: Optional[str] = None

    @property
    def fallback_used(self) -> Optional[str]:
        return self._fallback_used

    def query(self, question: str, top_k: int = 5,
              **kwargs) -> RAGResult:
        for strategy in self._strategies:
            if not strategy.enabled:
                continue
            for attempt in range(strategy.max_retries + 1):
                try:
                    result = self._execute_strategy(
                        strategy.name, question, top_k, **kwargs
                    )
                    self._fallback_used = strategy.name
                    return result
                except Exception as exc:
                    logger.debug(
                        "Strategy '%s' attempt %d failed: %s",
                        strategy.name, attempt + 1, exc,
                    )
                    if attempt < strategy.max_retries:
                        continue
                    break
        self._fallback_used = "none"
        return RAGResult(
            answer="I could not find an answer. The retrieval and fallback strategies all failed.",
            confidence=0.0,
        )

    def _execute_strategy(self, name: str, question: str, top_k: int,
                          **kwargs) -> RAGResult:
        if name == "primary_retriever":
            return self._pipeline.query(question, top_k=top_k, **kwargs)

        if name == "keyword_fallback":
            return self._keyword_fallback(question, top_k, **kwargs)

        if name == "llm_only":
            return self._llm_only_fallback(question, **kwargs)

        raise ValueError(f"Unknown fallback strategy: {name}")

    def _keyword_fallback(self, question: str, top_k: int,
                          **kwargs) -> RAGResult:
        chunks = self._pipeline._documents
        if not chunks:
            raise RuntimeError("No documents available for keyword search")

        keywords = set(question.lower().split())
        scored = []
        for doc in chunks:
            text_lower = doc.text.lower()
            score = sum(1 for kw in keywords if kw in text_lower) / max(len(keywords), 1)
            if score > 0:
                scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_chunks = [
            ChunkWithScore(text=d.text[:1000], score=s)
            for d, s in scored[:top_k]
        ]

        from runeextract.rag.types import RAGResult, Citation
        import time
        context = "\n".join(c.text for c in top_chunks)
        answer = self._pipeline.ai._call(
            "Answer using only the context below. If unsure, say so.",
            f"Context:\n{context}\n\nQuestion: {question}",
            **kwargs,
        )
        return RAGResult(
            answer=answer,
            retrieved_chunks=top_chunks,
            confidence=0.3,
        )

    def _llm_only_fallback(self, question: str, **kwargs) -> RAGResult:
        answer = self._pipeline.ai._call(
            "You are a helpful assistant. Answer the question using your own knowledge.",
            question,
            **kwargs,
        )
        return RAGResult(answer=answer, confidence=0.1)
