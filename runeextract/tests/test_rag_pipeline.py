"""
Tests for the RAG pipeline module (runeextract.rag).
"""

import json
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock

import pytest

from runeextract.rag.types import RAGResult, Citation, ChunkWithScore
from runeextract.rag.compressor import ContextualCompressor
from runeextract.rag.evaluate import RAGEvaluator
from runeextract.models.document import Document


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class TestRAGTypes:
    def test_rag_result_defaults(self):
        r = RAGResult(answer="Hello")
        assert r.answer == "Hello"
        assert r.citations == []
        assert r.confidence == 0.0
        assert r.latency_ms == 0.0

    def test_citation_fields(self):
        c = Citation(text="foo", source="bar.pdf", page=3, chunk_index=1, relevance_score=0.95)
        assert c.source == "bar.pdf"
        assert c.page == 3

    def test_chunk_with_score_defaults(self):
        c = ChunkWithScore(text="test", score=0.8)
        assert c.text == "test"
        assert c.metadata == {}


# ---------------------------------------------------------------------------
# ContextualCompressor
# ---------------------------------------------------------------------------

class TestContextualCompressor:
    def test_no_compression_needed(self):
        compressor = ContextualCompressor(max_tokens=5000)
        chunks = [
            ChunkWithScore(text="short text", score=0.9),
        ]
        result = compressor.compress(chunks, "test query", max_tokens=5000)
        assert len(result) == 1
        assert result[0].text == "short text"

    def test_merge_adjacent_chunks_same_source(self):
        compressor = ContextualCompressor(max_tokens=5000)
        chunks = [
            ChunkWithScore(text="First part", score=0.9, source="doc.pdf"),
            ChunkWithScore(text="Second part", score=0.8, source="doc.pdf"),
        ]
        result = compressor._merge_adjacent(chunks)
        assert len(result) == 1
        assert "First part" in result[0].text
        assert "Second part" in result[0].text

    def test_merge_adjacent_different_source(self):
        compressor = ContextualCompressor(max_tokens=5000)
        chunks = [
            ChunkWithScore(text="First", score=0.9, source="a.pdf"),
            ChunkWithScore(text="Second", score=0.8, source="b.pdf"),
        ]
        result = compressor._merge_adjacent(chunks)
        assert len(result) == 2

    def test_truncate_by_relevance(self):
        compressor = ContextualCompressor(max_tokens=5000)
        chunks = [
            ChunkWithScore(text="A " * 100, score=0.9),
            ChunkWithScore(text="B " * 100, score=0.5),
            ChunkWithScore(text="C " * 100, score=0.3),
        ]
        result = compressor._truncate_by_relevance(chunks, "query", budget=50)
        assert len(result) < len(chunks)
        # Highest scored chunks should be kept first
        assert all(c.score >= 0.5 for c in result)

    def test_estimate_tokens(self):
        assert ContextualCompressor._estimate_tokens("hello world") == 2  
        text = "word " * 100
        expected = int(100 * 1.33)
        assert ContextualCompressor._estimate_tokens(text) == expected


# ---------------------------------------------------------------------------
# RAGEvaluator
# ---------------------------------------------------------------------------

class TestRAGEvaluator:
    def test_generate_test_no_llm(self):
        evaluator = RAGEvaluator()
        docs = [Document(text="Hello world.", source_type="text")]
        tests = evaluator.generate_test_set(docs, num_questions=5)
        assert tests == []

    def test_generate_test_with_mock_llm(self):
        mock_llm = Mock(return_value="Q: What is this?\nA: Hello world.")
        evaluator = RAGEvaluator(llm_complete=mock_llm)
        docs = [Document(text="Hello world. This is a test document.", source_type="text")]
        tests = evaluator.generate_test_set(docs, num_questions=2)
        assert len(tests) <= 2
        if tests:
            assert "question" in tests[0]
            assert "answer" in tests[0]

    def test_evaluate_no_query_fn(self):
        evaluator = RAGEvaluator()
        result = evaluator.evaluate([{"question": "q", "answer": "a"}])
        assert "error" in result

    def test_parse_qa(self):
        evaluator = RAGEvaluator()
        q, a = evaluator._parse_qa("Q: What is X?\nA: X is Y.")
        assert q == "What is X?"
        assert a == "X is Y."

    def test_parse_qa_no_match(self):
        evaluator = RAGEvaluator()
        q, a = evaluator._parse_qa("some random text")
        assert q is None
        assert a is None

    def test_rate_relevance(self):
        evaluator = RAGEvaluator()
        score = evaluator._rate_relevance("The answer is 42.", "What is the answer?")
        assert 0.0 < score <= 1.0

    def test_has_answer(self):
        evaluator = RAGEvaluator()
        retrieved = ["The sky is blue.", "Grass is green."]
        score = evaluator._has_answer(retrieved, "sky")
        assert score == 0.5

    def test_rate_faithfulness(self):
        evaluator = RAGEvaluator()
        score = evaluator._rate_faithfulness("Blue sky.", "The sky is blue and clear today.")
        assert score >= 0.5

    def test_semantic_similarity(self):
        evaluator = RAGEvaluator()
        score = evaluator._semantic_similarity("hello world", "world hello")
        assert score == 1.0
        score2 = evaluator._semantic_similarity("hello world", "goodbye world")
        assert score2 < 1.0

    def test_aggregate(self):
        result = RAGEvaluator._aggregate([1.0, 2.0, 3.0])
        assert result["mean"] == 2.0
        assert result["count"] == 3
        assert result["min"] == 1.0
        assert result["max"] == 3.0

    def test_aggregate_empty(self):
        result = RAGEvaluator._aggregate([])
        assert result["count"] == 0

    def test_generate_test_with_mock_qa_missing_answer(self):
        mock_llm = Mock(return_value="Some random text without Q and A format.")
        evaluator = RAGEvaluator(llm_complete=mock_llm)
        docs = [Document(text="Hello world.", source_type="text")]
        tests = evaluator.generate_test_set(docs, num_questions=1)
        assert len(tests) == 0

    def test_evaluate_with_mock_query(self):
        mock_result = MagicMock()
        mock_result.answer = "The sky is blue."
        mock_result.retrieved_chunks = [
            ChunkWithScore(text="The sky is blue.", score=0.9, source="doc.txt")
        ]
        mock_result.citations = []

        mock_query_fn = Mock(return_value=mock_result)
        evaluator = RAGEvaluator(query_fn=mock_query_fn)
        test_set = [
            {"question": "What color is the sky?", "answer": "blue",
             "chunk_text": "The sky is blue.", "source": "doc.txt"}
        ]
        result = evaluator.evaluate(test_set)
        assert "answer_relevance" in result
        assert "context_precision" in result
        assert isinstance(result["answer_relevance"]["mean"], float)


# ---------------------------------------------------------------------------
# Lazy import from runeextract top level
# ---------------------------------------------------------------------------

class TestTopLevelExports:
    def test_auto_rag_function_importable(self):
        from runeextract import auto_rag
        assert callable(auto_rag)

    def test_auto_rag_class_importable(self):
        from runeextract import AutoRAG
        assert callable(AutoRAG)


# ---------------------------------------------------------------------------
# AutoRAG basic unit tests (mocked)
# ---------------------------------------------------------------------------

class TestAutoRAGBasic:
    def test_init_defaults(self):
        from runeextract.rag.auto_pipeline import AutoRAG
        rag = AutoRAG()
        assert rag.embedding_spec == "openai:text-embedding-3-small"
        assert rag.vector_store_type == "chromadb"
        assert rag.chunking_mode == "auto"

    def test_private_attributes_after_init(self):
        from runeextract.rag.auto_pipeline import AutoRAG
        rag = AutoRAG()
        assert rag._documents == []
        assert rag._retriever is None

    def test_resolve_source_single_file(self):
        from runeextract.rag.auto_pipeline import AutoRAG
        rag = AutoRAG()
        sources = rag._resolve_source("test.pdf")
        assert sources == ["test.pdf"]

    def test_resolve_source_list(self):
        from runeextract.rag.auto_pipeline import AutoRAG
        rag = AutoRAG()
        sources = rag._resolve_source(["a.pdf", "b.pdf"])
        assert sources == ["a.pdf", "b.pdf"]

    def test_deduplicate_identical_chunks(self):
        from runeextract.rag.auto_pipeline import AutoRAG
        rag = AutoRAG()
        chunks = [
            ChunkWithScore(text="Hello world. " * 10, score=0.9, source="a.pdf"),
            ChunkWithScore(text="Hello world. " * 10, score=0.8, source="a.pdf"),
            ChunkWithScore(text="Different text. " * 10, score=0.7, source="b.pdf"),
        ]
        result = rag._deduplicate(chunks)
        assert len(result) == 2  

    def test_compute_confidence_empty(self):
        from runeextract.rag.auto_pipeline import AutoRAG
        rag = AutoRAG()
        assert rag._compute_confidence([]) == 0.0

    def test_compute_confidence(self):
        from runeextract.rag.auto_pipeline import AutoRAG
        rag = AutoRAG()
        chunks = [
            ChunkWithScore(text="a", score=0.8),
            ChunkWithScore(text="b", score=0.6),
        ]
        assert rag._compute_confidence(chunks) == 0.7

    def test_resolve_chunking_auto_code_ext(self):
        from runeextract.rag.auto_pipeline import AutoRAG
        rag = AutoRAG(chunking="auto")
        doc = Document(text="print('hello')", source_type="code",
                       source_path="script.py")
        strategy = rag._resolve_chunking(doc)
        assert strategy == "by_heading"

    def test_resolve_chunking_auto_heading(self):
        from runeextract.rag.auto_pipeline import AutoRAG
        rag = AutoRAG(chunking="auto")
        doc = Document(text="## Introduction\nHello world.", source_type="text",
                       source_path="doc.md")
        strategy = rag._resolve_chunking(doc)
        assert strategy == "by_heading"

    def test_resolve_chunking_auto_long_doc(self):
        from runeextract.rag.auto_pipeline import AutoRAG
        rag = AutoRAG(chunking="auto")
        long_text = "word " * 20000
        doc = Document(text=long_text, source_type="text", source_path="long.txt")
        strategy = rag._resolve_chunking(doc)
        assert strategy == "hierarchical"

    def test_resolve_chunking_explicit(self):
        from runeextract.rag.auto_pipeline import AutoRAG
        rag = AutoRAG(chunking="fixed_size")
        doc = Document(text="hello", source_type="text", source_path="a.txt")
        strategy = rag._resolve_chunking(doc)
        assert strategy == "fixed_size"

    def test_resolve_chunking_fallback(self):
        from runeextract.rag.auto_pipeline import AutoRAG
        rag = AutoRAG(chunking="auto")
        doc = Document(text="Just a normal short sentence.", source_type="text",
                       source_path="doc.txt")
        strategy = rag._resolve_chunking(doc)
        assert strategy == "sentence_window"

    def test_auto_rag_factory_with_mock_extract(self):
        from runeextract.rag.auto_pipeline import auto_rag
        with patch("runeextract.rag.auto_pipeline.extract") as mock_extract:
            mock_extract.return_value = Document(
                text="Test document content.",
                source_type="text",
                source_path="test.txt",
            )
            rag = auto_rag("test.txt")
            assert rag._documents[0].text == "Test document content."


# ---------------------------------------------------------------------------
# Integration: AutoRAG with mocked AI
# ---------------------------------------------------------------------------

class TestAutoRAGQuery:
    def test_query_returns_rag_result(self):
        from runeextract.rag.auto_pipeline import AutoRAG

        rag = AutoRAG(llm="openai:gpt-4o-mini")

        mock_ai = MagicMock()
        mock_ai.embed.return_value = [[0.1] * 384]
        mock_ai._call.return_value = "The answer is 42. [1]"
        mock_ai._total_input_tokens = 50
        mock_ai._total_output_tokens = 20
        rag._ai = mock_ai

        mock_retriever = MagicMock()
        mock_retriever.query.return_value = [
            ChunkWithScore(text="The answer is 42.", score=0.95,
                           source="doc.txt", chunk_id="chunk_0"),
        ]
        rag._retriever = mock_retriever

        result = rag.query("What is the answer?", top_k=3)
        assert isinstance(result, RAGResult)
        assert "42" in result.answer
        assert result.latency_ms >= 0

    def test_query_with_multi_query(self):
        from runeextract.rag.auto_pipeline import AutoRAG

        rag = AutoRAG(llm="openai:gpt-4o-mini")

        mock_ai = MagicMock()
        mock_ai.embed.return_value = [[0.1] * 384]
        mock_ai._call.return_value = "The answer is 42. [1]"
        mock_ai.expand_query.return_value = ["Variant 1?", "Variant 2?"]
        mock_ai._total_input_tokens = 50
        mock_ai._total_output_tokens = 20
        rag._ai = mock_ai

        mock_retriever = MagicMock()
        mock_retriever.query.return_value = [
            ChunkWithScore(text="The answer is 42.", score=0.95,
                           source="doc.txt", chunk_id="chunk_0"),
        ]
        rag._retriever = mock_retriever

        result = rag.query("What is the answer?", multi_query=True)
        assert len(result.query_variants) == 2

    def test_query_with_hyde(self):
        from runeextract.rag.auto_pipeline import AutoRAG

        rag = AutoRAG(llm="openai:gpt-4o-mini")

        mock_ai = MagicMock()
        mock_ai.embed.return_value = [[0.1] * 384]
        mock_ai._call.return_value = "The expected answer is 42. [1]"
        mock_ai.hyde.return_value = "A hypothetical document about the meaning of life."
        mock_ai._total_input_tokens = 50
        mock_ai._total_output_tokens = 20
        rag._ai = mock_ai

        mock_retriever = MagicMock()
        mock_retriever.query.return_value = [
            ChunkWithScore(text="The answer is 42.", score=0.95,
                           source="doc.txt", chunk_id="chunk_0"),
        ]
        rag._retriever = mock_retriever

        result = rag.query("What is the answer?", hyde=True)
        assert "42" in result.answer

    def test_query_answer_length(self):
        from runeextract.rag.auto_pipeline import AutoRAG

        rag = AutoRAG(llm="openai:gpt-4o-mini")
        mock_ai = MagicMock()
        mock_ai.embed.return_value = [[0.1] * 384]
        mock_ai._call.return_value = "Short answer. [1]"
        mock_ai._total_input_tokens = 50
        mock_ai._total_output_tokens = 20
        rag._ai = mock_ai

        mock_retriever = MagicMock()
        mock_retriever.query.return_value = [
            ChunkWithScore(text="Some content.", score=0.9, source="doc.txt"),
        ]
        rag._retriever = mock_retriever

        result = rag.query("Test?", answer_length="short")
        assert result.answer is not None


# ---------------------------------------------------------------------------
# ChromaRetriever basic tests (import only)
# ---------------------------------------------------------------------------

class TestRetrieverImports:
    def test_chroma_retriever_import(self):
        from runeextract.rag.retriever import ChromaRetriever
        assert ChromaRetriever is not None

    def test_faiss_retriever_import(self):
        from runeextract.rag.retriever import FAISSRetriever
        assert FAISSRetriever is not None
