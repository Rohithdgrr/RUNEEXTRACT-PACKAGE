"""
Tests for RAG Phase 2 + 4: query_router, hybrid_search, context_packer,
robust_rag, confidence, debugger.
"""

import json
from unittest.mock import Mock, patch, MagicMock

import pytest

from runeextract.rag.types import ChunkWithScore


# ---------------------------------------------------------------------------
# QueryRouter
# ---------------------------------------------------------------------------

class TestQueryRouter:
    def test_classify_factual_default(self):
        from runeextract.rag.query_router import QueryRouter, QueryIntent
        router = QueryRouter()
        assert router.classify("What is the capital of France?") == QueryIntent.FACTUAL

    def test_classify_comparative(self):
        from runeextract.rag.query_router import QueryRouter, QueryIntent
        router = QueryRouter()
        assert router.classify("Compare Q1 and Q2 results") == QueryIntent.COMPARATIVE

    def test_classify_analytical(self):
        from runeextract.rag.query_router import QueryRouter, QueryIntent
        router = QueryRouter()
        assert router.classify("Why did revenue decline in 2024?") == QueryIntent.ANALYTICAL

    def test_classify_summarization(self):
        from runeextract.rag.query_router import QueryRouter, QueryIntent
        router = QueryRouter()
        assert router.classify("Summarize the key findings") == QueryIntent.SUMMARIZATION

    def test_classify_exploratory(self):
        from runeextract.rag.query_router import QueryRouter, QueryIntent
        router = QueryRouter()
        assert router.classify("List the types of documents") == QueryIntent.EXPLORATORY

    def test_extract_filters_year(self):
        from runeextract.rag.query_router import QueryRouter
        router = QueryRouter()
        filters = router.extract_filters("revenue in 2024")
        assert filters.get("year") == "2024"

    def test_extract_filters_year_range(self):
        from runeextract.rag.query_router import QueryRouter
        router = QueryRouter()
        filters = router.extract_filters("results from 2020 to 2024")
        assert filters.get("year_range") == "2020-2024"

    def test_extract_filters_author(self):
        from runeextract.rag.query_router import QueryRouter
        router = QueryRouter()
        filters = router.extract_filters('by "John Smith"')
        assert filters.get("author") == "John Smith"

    def test_extract_filters_section(self):
        from runeextract.rag.query_router import QueryRouter
        router = QueryRouter()
        filters = router.extract_filters("in section 5")
        assert filters.get("section") == "5"

    def test_decompose_rule_based(self):
        from runeextract.rag.query_router import QueryRouter
        router = QueryRouter()
        dq = router.decompose("Compare Q1 and Q2 results")
        assert len(dq.sub_queries) >= 1
        assert dq.intent.value == "comparative"

    def test_decompose_with_llm(self):
        from runeextract.rag.query_router import QueryRouter
        mock_llm = Mock(return_value="What is Q1 revenue?\nWhat is Q2 revenue?")
        router = QueryRouter(llm_complete=mock_llm)
        dq = router.decompose("Compare Q1 and Q2 revenue")
        assert len(dq.sub_queries) >= 1
        assert dq.sub_queries[0] != ""

    def test_decompose_empty_llm_response(self):
        from runeextract.rag.query_router import QueryRouter
        mock_llm = Mock(return_value="")
        router = QueryRouter(llm_complete=mock_llm)
        dq = router.decompose("simple question")
        assert len(dq.sub_queries) == 1


# ---------------------------------------------------------------------------
# BM25Sparse
# ---------------------------------------------------------------------------

class TestBM25Sparse:
    def test_score_known_terms(self):
        from runeextract.rag.hybrid_search import BM25Sparse
        corpus = ["the cat sat on the mat", "the dog chased the cat"]
        bm25 = BM25Sparse(corpus)
        score = bm25.score("cat", 0)
        assert score > 0

    def test_score_unknown_terms(self):
        from runeextract.rag.hybrid_search import BM25Sparse
        corpus = ["hello world"]
        bm25 = BM25Sparse(corpus)
        score = bm25.score("zzzzz", 0)
        assert score == 0.0

    def test_tokenize(self):
        from runeextract.rag.hybrid_search import BM25Sparse
        assert BM25Sparse._tokenize("Hello, World!") == ["hello", "world"]


# ---------------------------------------------------------------------------
# HybridSearch
# ---------------------------------------------------------------------------

class TestHybridSearch:
    def test_analyze_query(self):
        from runeextract.rag.hybrid_search import HybridSearch
        hs = HybridSearch(dense_fn=lambda q: [0.1, 0.2])
        analysis = hs.analyze_query("what is the capital")
        assert "lexical_density" in analysis
        assert analysis["term_count"] == 4

    def test_analyze_query_empty(self):
        from runeextract.rag.hybrid_search import HybridSearch
        hs = HybridSearch(dense_fn=lambda q: [])
        analysis = hs.analyze_query("")
        assert analysis["lexical_density"] == 0.5

    def test_compute_weights(self):
        from runeextract.rag.hybrid_search import HybridSearch
        hs = HybridSearch(dense_fn=lambda q: [0.1])
        dense, sparse = hs.compute_weights("compare Q1 Q2 Q3 results")
        assert 0.0 <= dense <= 1.0
        assert 0.0 <= sparse <= 1.0
        assert abs(dense + sparse - 1.0) < 0.001

    def test_update_corpus(self):
        from runeextract.rag.hybrid_search import HybridSearch
        hs = HybridSearch(dense_fn=lambda q: [0.1])
        chunks = [ChunkWithScore(text="hello world", score=0.5)]
        hs.update_corpus(chunks)
        assert hs._corpus == ["hello world"]

    def test_reciprocal_rank_fusion(self):
        from runeextract.rag.hybrid_search import HybridSearch
        a = ChunkWithScore(text="doc a", score=0.9)
        b = ChunkWithScore(text="doc b", score=0.8)
        fused = HybridSearch.reciprocal_rank_fusion([a], [b], k=60, top_k=2)
        assert len(fused) == 2


# ---------------------------------------------------------------------------
# ContextPacker
# ---------------------------------------------------------------------------

class TestContextPacker:
    def test_pack_sorted(self):
        from runeextract.rag.context_packer import ContextPacker
        packer = ContextPacker(max_tokens=1000)
        chunks = [
            ChunkWithScore(text="A " * 200, score=0.9),
            ChunkWithScore(text="B " * 200, score=0.5),
        ]
        result = packer.pack(chunks, "query", strategy="sorted")
        assert result.chunks_used >= 1
        assert result.total_tokens <= 1000

    def test_pack_compressed(self):
        from runeextract.rag.context_packer import ContextPacker
        packer = ContextPacker(max_tokens=500)
        chunks = [
            ChunkWithScore(text="High relevance " * 100, score=0.9),
            ChunkWithScore(text="Low relevance " * 100, score=0.2),
        ]
        result = packer.pack(chunks, "query", strategy="compressed")
        assert result.chunks_used >= 1

    def test_pack_structured(self):
        from runeextract.rag.context_packer import ContextPacker
        packer = ContextPacker(max_tokens=2000)
        chunks = [
            ChunkWithScore(text="Doc1 text " * 50, score=0.9, source="doc1.pdf"),
            ChunkWithScore(text="Doc2 text " * 50, score=0.8, source="doc2.pdf"),
        ]
        result = packer.pack(chunks, "query", strategy="structured")
        assert result.chunks_used >= 1
        assert result.strategy == "structured"

    def test_pack_budget_allows_all(self):
        from runeextract.rag.context_packer import ContextPacker
        packer = ContextPacker(max_tokens=10000)
        chunks = [
            ChunkWithScore(text="short", score=0.9),
            ChunkWithScore(text="text", score=0.8),
        ]
        result = packer.pack(chunks, "query", strategy="sorted")
        assert result.chunks_used == 2

    def test_pack_empty_chunks(self):
        from runeextract.rag.context_packer import ContextPacker
        packer = ContextPacker()
        result = packer.pack([], "query")
        assert result.chunks_used == 0
        assert result.text == ""

    def test_unknown_strategy_falls_back_to_sorted(self):
        from runeextract.rag.context_packer import ContextPacker
        packer = ContextPacker()
        chunks = [ChunkWithScore(text="test", score=0.5)]
        result = packer.pack(chunks, "query", strategy="nonexistent")
        assert result.chunks_used >= 1


# ---------------------------------------------------------------------------
# RobustRAG
# ---------------------------------------------------------------------------

class TestRobustRAG:
    def test_query_success(self):
        from runeextract.rag.robust_rag import RobustRAG
        from runeextract.rag.types import RAGResult

        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = RAGResult(answer="42", confidence=0.9)
        mock_pipeline._documents = []

        rag = RobustRAG(mock_pipeline)
        result = rag.query("What?")
        assert result.answer == "42"
        assert rag.fallback_used == "primary_retriever"

    def test_query_fallback_keyword(self):
        from runeextract.rag.robust_rag import RobustRAG
        from runeextract.rag.types import RAGResult

        mock_pipeline = MagicMock()
        mock_pipeline.query.side_effect = RuntimeError("retriever failed")
        mock_doc = MagicMock()
        mock_doc.text = "the answer is 42"
        mock_doc.source_path = "test.txt"
        mock_pipeline._documents = [mock_doc]
        mock_ai = MagicMock()
        mock_ai._call.return_value = "42"
        mock_pipeline.ai = mock_ai

        rag = RobustRAG(mock_pipeline)
        result = rag.query("what answer")
        assert result.answer is not None
        assert "42" in result.answer

    def test_query_all_fail(self):
        from runeextract.rag.robust_rag import RobustRAG
        from runeextract.rag.types import RAGResult

        mock_pipeline = MagicMock()
        mock_pipeline.query.side_effect = RuntimeError("fail")
        mock_pipeline._documents = []
        mock_pipeline.ai._call.side_effect = RuntimeError("llm fail")

        rag = RobustRAG(mock_pipeline)
        result = rag.query("What?")
        assert "could not find" in result.answer.lower()
        assert rag.fallback_used == "none"

    def test_custom_strategies(self):
        from runeextract.rag.robust_rag import RobustRAG, FallbackStrategy

        mock_pipeline = MagicMock()
        mock_pipeline.query.side_effect = RuntimeError("fail")
        mock_pipeline._documents = []

        rag = RobustRAG(mock_pipeline, strategies=[
            FallbackStrategy("primary_retriever", max_retries=0),
            FallbackStrategy("llm_only"),
        ])
        result = rag.query("What?")
        assert rag.fallback_used is not None

    def test_fallback_property(self):
        from runeextract.rag.robust_rag import RobustRAG

        mock_pipeline = MagicMock()
        mock_pipeline.query.side_effect = RuntimeError("fail")
        mock_pipeline._documents = []

        rag = RobustRAG(mock_pipeline)
        result = rag.query("What?")
        assert rag.fallback_used is not None


# ---------------------------------------------------------------------------
# ConfidenceScorer
# ---------------------------------------------------------------------------

class TestConfidenceScorer:
    def test_score_empty_chunks(self):
        from runeextract.rag.confidence import ConfidenceScorer
        scorer = ConfidenceScorer()
        factors = scorer.score([], "answer", "question")
        assert factors.overall == 0.0

    def test_retrieval_confidence(self):
        from runeextract.rag.confidence import ConfidenceScorer
        scorer = ConfidenceScorer()
        chunks = [
            ChunkWithScore(text="a", score=0.9),
            ChunkWithScore(text="b", score=0.7),
        ]
        score = scorer._retrieval_confidence(chunks)
        assert 0.0 <= score <= 1.0

    def test_source_diversity(self):
        from runeextract.rag.confidence import ConfidenceScorer
        chunks = [
            ChunkWithScore(text="a", score=0.9, source="doc1.pdf"),
            ChunkWithScore(text="b", score=0.8, source="doc2.pdf"),
        ]
        score = ConfidenceScorer._source_diversity(chunks)
        assert score > 0.3

    def test_chunk_relevance(self):
        from runeextract.rag.confidence import ConfidenceScorer
        chunks = [
            ChunkWithScore(text="revenue grew 20% in 2024", score=0.9),
        ]
        score = ConfidenceScorer._chunk_relevance(chunks, "revenue 2024")
        assert score > 0

    def test_lexical_faithfulness(self):
        from runeextract.rag.confidence import ConfidenceScorer
        chunks = [ChunkWithScore(text="the sky is blue", score=0.9)]
        score = ConfidenceScorer._lexical_faithfulness("sky blue", chunks)
        assert score > 0

    def test_faithfulness_with_llm_judge(self):
        from runeextract.rag.confidence import ConfidenceScorer
        mock_judge = Mock(return_value=0.85)
        scorer = ConfidenceScorer(llm_judge=mock_judge)
        chunks = [ChunkWithScore(text="test", score=0.9)]
        score = scorer._faithfulness("answer text", chunks)
        assert score == 0.85

    def test_overall_score(self):
        from runeextract.rag.confidence import ConfidenceScorer
        scorer = ConfidenceScorer()
        chunks = [
            ChunkWithScore(text="revenue up 20% in Q1 2024", score=0.9, source="doc1.pdf"),
            ChunkWithScore(text="revenue grew in 2024", score=0.8, source="doc2.pdf"),
        ]
        factors = scorer.score(chunks, "Revenue grew 20%.", "What happened to revenue?")
        assert factors.overall > 0
        assert factors.retrieval_score > 0
        assert factors.source_diversity > 0


# ---------------------------------------------------------------------------
# RAGDebugger
# ---------------------------------------------------------------------------

class TestRAGDebugger:
    def test_trace_returns_debug_trace(self):
        from runeextract.rag.debugger import RAGDebugger, DebugTrace

        mock_pipeline = MagicMock()
        mock_pipeline.ai.expand_query.return_value = ["V1?", "V2?"]
        mock_pipeline.ai.hyde.return_value = "Hypothetical doc."
        mock_pipeline.ai.rerank.return_value = [("text", 0.9)]
        mock_pipeline._retrieve.return_value = [
            ChunkWithScore(text="chunk text", score=0.9, source="doc.txt")
        ]
        mock_pipeline._deduplicate.side_effect = lambda x: x
        mock_pipeline._compressor.compress.side_effect = lambda c, q: c
        mock_pipeline._generate_answer.return_value = ("The answer is 42.", [])
        mock_pipeline.reranker_spec = "some-reranker"

        debugger = RAGDebugger(mock_pipeline)
        trace = debugger.trace("What is the answer?", multi_query=True, hyde=True)
        assert isinstance(trace, DebugTrace)
        assert trace.query == "What is the answer?"
        assert len(trace.query_variants) >= 1
        assert trace.answer is not None
        assert trace.latency_ms >= 0
        assert "query_expansion" in trace.stages
        assert "retrieval" in trace.stages
        assert "reranking" in trace.stages

    def test_print_trace_no_crash(self):
        from runeextract.rag.debugger import RAGDebugger, DebugTrace
        trace = DebugTrace(
            query="test?",
            answer="yes",
            query_variants=["var1"],
            retrieved_chunks=[ChunkWithScore(text="chunk", score=0.9)],
            reranked_chunks=[ChunkWithScore(text="chunk", score=0.9)],
            compressed_chunks=[ChunkWithScore(text="chunk", score=0.9)],
            constructed_prompt="prompt text",
            stages={"retrieval": 10.0},
            latency_ms=100.0,
        )
        RAGDebugger.print_trace(trace)

    def test_trace_to_dict(self):
        from runeextract.rag.debugger import RAGDebugger, DebugTrace
        trace = DebugTrace(query="q", answer="a")
        d = RAGDebugger(None).trace_to_dict(trace)
        assert d["query"] == "q"
        assert d["answer"] == "a"
        assert "latency_ms" in d
        assert "stages" in d

    def test_trace_errors_on_failure(self):
        from runeextract.rag.debugger import RAGDebugger

        mock_pipeline = MagicMock()
        mock_pipeline.ai.expand_query.side_effect = RuntimeError("expansion failed")
        mock_pipeline._retrieve.side_effect = RuntimeError("retrieve failed")
        mock_pipeline._deduplicate.side_effect = lambda x: x
        mock_pipeline._compressor.compress.side_effect = lambda c, q: c
        mock_pipeline._generate_answer.return_value = ("fallback", [])
        mock_pipeline.reranker_spec = None

        debugger = RAGDebugger(mock_pipeline)
        trace = debugger.trace("What?", multi_query=True)
        assert len(trace.errors) > 0
        assert any("expansion failed" in e for e in trace.errors)


# ---------------------------------------------------------------------------
# Top-level imports
# ---------------------------------------------------------------------------

class TestTopLevelExports:
    def test_robust_rag_importable(self):
        from runeextract import RobustRAG
        assert callable(RobustRAG)

    def test_rag_debugger_importable(self):
        from runeextract import RAGDebugger
        assert callable(RAGDebugger)

    def test_confidence_scorer_importable(self):
        from runeextract import ConfidenceScorer
        assert callable(ConfidenceScorer)

    def test_query_router_importable(self):
        from runeextract import QueryRouter
        assert callable(QueryRouter)

    def test_rag_submodule_imports(self):
        from runeextract.rag import (
            QueryRouter, QueryIntent, DecomposedQuery,
            HybridSearch, HybridResult, BM25Sparse,
            ContextPacker, PackedContext,
            RobustRAG, FallbackStrategy,
            ConfidenceScorer, ConfidenceFactors,
            RAGDebugger, DebugTrace,
        )
        assert QueryRouter is not None
