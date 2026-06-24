"""
RAG pipeline step-by-step debugger.

``RAGDebugger`` wraps an ``AutoRAG`` query and records every stage of
the pipeline — query expansion, retrieval, reranking, compression,
prompt construction, and answer generation — for inspection.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from runeextract.rag.auto_pipeline import AutoRAG
from runeextract.rag.types import RAGResult, ChunkWithScore

logger = logging.getLogger(__name__)


@dataclass
class DebugTrace:
    query: str = ""
    query_variants: List[str] = field(default_factory=list)
    retrieved_chunks: List[ChunkWithScore] = field(default_factory=list)
    reranked_chunks: List[ChunkWithScore] = field(default_factory=list)
    compressed_chunks: List[ChunkWithScore] = field(default_factory=list)
    constructed_prompt: str = ""
    answer: str = ""
    latency_ms: float = 0.0
    stages: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class RAGDebugger:
    """Wrap an AutoRAG instance to produce detailed execution traces.

    Usage::

        rag = AutoRAG()
        rag.ingest("docs.pdf")
        debugger = RAGDebugger(rag)
        trace = debugger.trace("What is X?")
        debugger.print_trace(trace)
    """

    def __init__(self, pipeline: AutoRAG):
        self._pipeline = pipeline

    def trace(self, question: str, top_k: int = 5,
              hyde: bool = False,
              multi_query: bool = False,
              **kwargs) -> DebugTrace:
        trace = DebugTrace(query=question)
        start = time.time()

        # 1. Query expansion
        t0 = time.time()
        queries = [question]
        if multi_query:
            try:
                queries.extend(self._pipeline.ai.expand_query(question))
                trace.query_variants = queries[1:]
            except Exception as exc:
                trace.errors.append(f"Query expansion: {exc}")
        if hyde:
            try:
                queries.append(self._pipeline.ai.hyde(question))
                if len(queries) > 1:
                    trace.query_variants = queries[1:]
            except Exception as exc:
                trace.errors.append(f"HyDE: {exc}")
        trace.stages["query_expansion"] = (time.time() - t0) * 1000

        # 2. Retrieval
        t0 = time.time()
        all_chunks: List[ChunkWithScore] = []
        for q in queries:
            try:
                chunks = self._pipeline._retrieve(q, top_k=top_k * 2)
                all_chunks.extend(chunks)
            except Exception as exc:
                trace.errors.append(f"Retrieval for '{q[:50]}': {exc}")
        trace.retrieved_chunks = self._pipeline._deduplicate(all_chunks)
        trace.stages["retrieval"] = (time.time() - t0) * 1000

        # 3. Reranking
        t0 = time.time()
        unique = list(trace.retrieved_chunks)
        if self._pipeline.reranker_spec and len(unique) > 1:
            try:
                texts = [c.text for c in unique]
                reranked = self._pipeline.ai.rerank(question, texts, top_k=top_k)
                seen = set()
                ranked = []
                for text, score in reranked:
                    for c in unique:
                        if c.text == text and id(c) not in seen:
                            c.score = score
                            seen.add(id(c))
                            ranked.append(c)
                            break
                trace.reranked_chunks = ranked[:top_k]
            except Exception as exc:
                trace.errors.append(f"Reranking: {exc}")
                trace.reranked_chunks = unique[:top_k]
        else:
            trace.reranked_chunks = unique[:top_k]
        trace.stages["reranking"] = (time.time() - t0) * 1000

        # 4. Compression
        t0 = time.time()
        try:
            trace.compressed_chunks = self._pipeline._compressor.compress(
                trace.reranked_chunks, question
            )
        except Exception as exc:
            trace.errors.append(f"Compression: {exc}")
            trace.compressed_chunks = trace.reranked_chunks
        trace.stages["compression"] = (time.time() - t0) * 1000

        # 5. Prompt construction
        t0 = time.time()
        answer, _ = self._pipeline._generate_answer(
            question, trace.compressed_chunks, return_citations=True,
            length=kwargs.get("answer_length", "medium"),
        )
        trace.constructed_prompt = self._build_prompt_debug(
            question, trace.compressed_chunks
        )
        trace.answer = answer
        trace.stages["generation"] = (time.time() - t0) * 1000

        trace.latency_ms = (time.time() - start) * 1000
        return trace

    @staticmethod
    def _build_prompt_debug(question: str, chunks: List[ChunkWithScore]) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            src = c.source or "unknown"
            parts.append(f"[{i}] Source: {src}\n{c.text}")
        context = "\n---\n".join(parts)
        return (
            f"Answer the question using ONLY the provided context. "
            f"Cite sources using [1], [2], etc. "
            f"If the context doesn't contain the answer, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )

    @staticmethod
    def print_trace(trace: DebugTrace) -> None:
        """Pretty-print a debug trace to the console."""
        line = "=" * 60
        print(line)
        print(f"RAG DEBUG TRACE")
        print(f"Query: {trace.query}")
        print(f"Latency: {trace.latency_ms:.1f}ms")
        print(line)

        print(f"\n--- Stages ---")
        for name, ms in trace.stages.items():
            print(f"  {name}: {ms:.1f}ms")

        if trace.query_variants:
            print(f"\n--- Query Variants ({len(trace.query_variants)}) ---")
            for v in trace.query_variants:
                print(f"  • {v[:120]}")

        print(f"\n--- Retrieved Chunks ({len(trace.retrieved_chunks)}) ---")
        for i, c in enumerate(trace.retrieved_chunks[:5], 1):
            print(f"  [{i}] score={c.score:.3f}  src={c.source}  text={c.text[:80]}...")
        if len(trace.retrieved_chunks) > 5:
            print(f"  ... and {len(trace.retrieved_chunks) - 5} more")

        print(f"\n--- Reranked Chunks ({len(trace.reranked_chunks)}) ---")
        for i, c in enumerate(trace.reranked_chunks, 1):
            print(f"  [{i}] score={c.score:.3f}  src={c.source}  text={c.text[:80]}...")

        print(f"\n--- Compressed Chunks ({len(trace.compressed_chunks)}) ---")
        for i, c in enumerate(trace.compressed_chunks, 1):
            print(f"  [{i}] score={c.score:.3f}  {c.text[:80]}...")

        print(f"\n--- Prompt (first 200 chars) ---")
        print(f"  {trace.constructed_prompt[:200]}...")

        print(f"\n--- Answer ---")
        print(f"  {trace.answer[:500]}")

        if trace.errors:
            print(f"\n--- Errors ---")
            for e in trace.errors:
                print(f"  ⚠ {e}")

        print(line)

    def trace_to_dict(self, trace: DebugTrace) -> Dict[str, Any]:
        """Export a trace as a JSON-serialisable dictionary."""
        return {
            "query": trace.query,
            "query_variants": trace.query_variants,
            "latency_ms": trace.latency_ms,
            "stages": trace.stages,
            "errors": trace.errors,
            "retrieved_chunks": [
                {"text": c.text[:200], "score": c.score, "source": c.source}
                for c in trace.retrieved_chunks
            ],
            "reranked_chunks": [
                {"text": c.text[:200], "score": c.score, "source": c.source}
                for c in trace.reranked_chunks
            ],
            "compressed_chunks": [
                {"text": c.text[:200], "score": c.score}
                for c in trace.compressed_chunks
            ],
            "answer": trace.answer,
        }
