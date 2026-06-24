"""
Tests for the RAG pipeline module (runeextract.rag).
"""

import json
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock

import pytest

from runeextract.rag.auto_pipeline import AutoRAG
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

class TestRAGEvaluatorEvaluate:
    def test_evaluate_with_llm_metrics(self):
        from runeextract.rag.evaluate import RAGEvaluator
        from runeextract.rag.types import RAGResult, ChunkWithScore

        mock_llm = Mock(return_value="0.85")
        mock_result = MagicMock()
        mock_result.answer = "Revenue grew 20% in Q1 2024."
        mock_result.retrieved_chunks = [
            ChunkWithScore(text="Revenue grew 20% in Q1 2024 according to the report.",
                           score=0.9, source="doc.txt")
        ]
        mock_result.citations = []

        evaluator = RAGEvaluator(query_fn=Mock(return_value=mock_result),
                                 llm_complete=mock_llm)
        test_set = [
            {"question": "What happened to revenue?", "answer": "grew 20%",
             "chunk_text": "Revenue grew 20%.", "source": "doc.txt"}
        ]
        result = evaluator.evaluate(test_set)
        assert "faithfulness_llm" in result
        assert "answer_relevance_llm" in result
        assert isinstance(result["faithfulness_llm"]["mean"], float)

    def test_evaluate_no_llm_faithfulness_fallback(self):
        from runeextract.rag.evaluate import RAGEvaluator
        from runeextract.rag.types import RAGResult, ChunkWithScore

        mock_result = MagicMock()
        mock_result.answer = "blue sky"
        mock_result.retrieved_chunks = [
            ChunkWithScore(text="The sky is blue and clear.", score=0.9, source="doc.txt")
        ]
        mock_result.citations = []

        evaluator = RAGEvaluator(query_fn=Mock(return_value=mock_result))
        test_set = [
            {"question": "What color?", "answer": "blue",
             "chunk_text": "The sky is blue.", "source": "doc.txt"}
        ]
        result = evaluator.evaluate(test_set)
        assert "faithfulness_llm" in result
        assert result["faithfulness_llm"]["mean"] <= 1.0

    def test_rate_relevance_llm_with_llm(self):
        from runeextract.rag.evaluate import RAGEvaluator
        mock_llm = Mock(return_value="0.9")
        evaluator = RAGEvaluator(llm_complete=mock_llm)
        score = evaluator._rate_relevance_llm("The sky is blue.", "What color?")
        assert score == 0.9

    def test_rate_relevance_llm_no_llm(self):
        from runeextract.rag.evaluate import RAGEvaluator
        evaluator = RAGEvaluator()
        score = evaluator._rate_relevance_llm("test", "question")
        assert score == 0.0

    def test_rate_faithfulness_llm_with_llm(self):
        from runeextract.rag.evaluate import RAGEvaluator
        mock_llm = Mock(return_value="0.95")
        evaluator = RAGEvaluator(llm_complete=mock_llm)
        score = evaluator._rate_faithfulness_llm("answer", "context that supports answer")
        assert score == 0.95

    def test_rate_faithfulness_llm_fallback_on_error(self):
        from runeextract.rag.evaluate import RAGEvaluator
        mock_llm = Mock(side_effect=ValueError("parse failed"))
        evaluator = RAGEvaluator(llm_complete=mock_llm)
        score = evaluator._rate_faithfulness_llm("blue sky", "the sky is blue")
        assert score > 0


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
        conf = rag._compute_confidence(chunks)
        assert 0.0 <= conf <= 1.0

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


class TestAutoRAGEnhanced:
    def test_generate_answer_with_max_tokens(self):
        from runeextract.rag.auto_pipeline import AutoRAG
        rag = AutoRAG()
        mock_ai = MagicMock()
        mock_ai._call.return_value = "Answer. [1]"
        rag._ai = mock_ai

        chunks = [
            ChunkWithScore(text="A " * 500, score=0.9, source="doc.txt"),
            ChunkWithScore(text="B " * 500, score=0.8, source="doc.txt"),
        ]
        answer, citations = rag._generate_answer(
            "question?", chunks, return_citations=True,
            length="short", max_tokens=200,
        )
        assert answer == "Answer. [1]"

    def test_generate_answer_no_max_tokens(self):
        from runeextract.rag.auto_pipeline import AutoRAG
        rag = AutoRAG()
        mock_ai = MagicMock()
        mock_ai._call.return_value = "Answer. [1]"
        rag._ai = mock_ai

        chunks = [ChunkWithScore(text="test", score=0.9, source="doc.txt")]
        answer, citations = rag._generate_answer(
            "question?", chunks, return_citations=True, length="short"
        )
        assert answer == "Answer. [1]"

    def test_build_context(self):
        from runeextract.rag.auto_pipeline import AutoRAG
        chunks = [
            ChunkWithScore(text="First", score=0.9, source="a.pdf", page=1),
            ChunkWithScore(text="Second", score=0.8, source="b.pdf"),
        ]
        ctx = AutoRAG._build_context(chunks)
        assert "[1]" in ctx
        assert "[2]" in ctx
        assert "p.1" in ctx

    def test_compute_confidence_enhanced(self):
        from runeextract.rag.auto_pipeline import AutoRAG
        rag = AutoRAG()
        chunks = [
            ChunkWithScore(text="revenue up 20%", score=0.9, source="doc1.pdf"),
            ChunkWithScore(text="revenue grew", score=0.7, source="doc2.pdf"),
        ]
        conf = rag._compute_confidence(chunks)
        assert 0.0 <= conf <= 1.0

    def test_compute_confidence_empty(self):
        from runeextract.rag.auto_pipeline import AutoRAG
        rag = AutoRAG()
        assert rag._compute_confidence([]) == 0.0


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


# ---------------------------------------------------------------------------
# Feature 2: Live Sync — _remove_from_index
# ---------------------------------------------------------------------------

class TestRemoveFromIndex:
    def test_removes_file_chunk_ids(self):
        rag = AutoRAG()
        rag._ai = MagicMock()
        rag._file_chunk_ids["/test/doc.pdf"] = ["chunk1", "chunk2"]
        rag._file_hashes["/test/doc.pdf"] = "abc123"
        rag._remove_from_index("/test/doc.pdf")
        assert "/test/doc.pdf" not in rag._file_chunk_ids
        assert "/test/doc.pdf" not in rag._file_hashes

    def test_skips_unknown_source(self):
        rag = AutoRAG()
        rag._ai = MagicMock()
        rag._file_chunk_ids["/other/doc.pdf"] = ["chunk1"]
        rag._remove_from_index("/nonexistent/doc.pdf")
        assert "/other/doc.pdf" in rag._file_chunk_ids

    def test_calls_delete_by_source_on_chroma(self):
        from runeextract.rag.retriever import ChromaRetriever
        rag = AutoRAG()
        rag._ai = MagicMock()
        rag.vector_store_type = "chromadb"
        mock_retriever = MagicMock(spec=ChromaRetriever)
        mock_retriever.delete_by_source.return_value = 3
        rag._retriever = mock_retriever
        rag._file_chunk_ids["/test/doc.pdf"] = ["c1", "c2", "c3"]
        result = rag._remove_from_index("/test/doc.pdf")
        mock_retriever.delete_by_source.assert_called_once_with("/test/doc.pdf")
        assert result == 3

    def test_tolerates_retriever_exception(self):
        from runeextract.rag.retriever import ChromaRetriever
        rag = AutoRAG()
        rag._ai = MagicMock()
        rag.vector_store_type = "chromadb"
        mock_retriever = MagicMock(spec=ChromaRetriever)
        mock_retriever.delete_by_source.side_effect = Exception("DB error")
        rag._retriever = mock_retriever
        rag._file_chunk_ids["/test/doc.pdf"] = ["c1"]
        rag._remove_from_index("/test/doc.pdf")
        assert "/test/doc.pdf" not in rag._file_chunk_ids


# ---------------------------------------------------------------------------
# Feature 4: Multi-Modal RAG
# ---------------------------------------------------------------------------

class TestMultiModalRAG:
    def test_multimodal_index_built_during_ingest(self):
        rag = AutoRAG(multimodal=True)
        rag._ai = MagicMock()
        rag._ai.embed.return_value = [[0.1, 0.2]]
        doc = Document(text="hello")
        doc.images = [MagicMock(data=b"fakeimg", caption="chart", format="png")]
        doc.tables = []
        rag._chunk_and_index([doc])
        assert rag._mm_index is not None
        assert rag._mm_index.item_count > 0

    def test_retrieve_fuses_multimodal_results(self):
        rag = AutoRAG(multimodal=True)
        rag._ai = MagicMock()
        rag._ai.embed.return_value = [[0.1, 0.2]]
        rag._mm_index = MagicMock()
        rag._mm_index.search.return_value = MagicMock(items=[])
        rag._retriever = MagicMock()
        rag._retriever.query.return_value = [
            ChunkWithScore(text="text chunk", score=0.9, source="doc.txt"),
        ]
        results = rag._retrieve("test query", multimodal=True)
        assert len(results) == 1
        assert results[0].text == "text chunk"

    def test_generate_answer_includes_images_in_context(self):
        rag = AutoRAG()
        mock_ai = MagicMock()
        mock_ai._call.return_value = "Answer with image context."
        rag._ai = mock_ai
        chunks = [ChunkWithScore(text="test", score=0.5, source="doc.txt")]
        images = [
            {"data": "base64data", "format": "png", "text": "chart showing growth", "source": "doc.pdf"},
        ]
        answer, citations = rag._generate_answer(
            "question?", chunks, return_citations=False,
            length="short", mm_images=images,
        )
        assert answer == "Answer with image context."

    def test_multimodal_images_in_result(self):
        rag = AutoRAG()
        rag._ai = MagicMock()
        rag._ai._call.return_value = "Answer. [1]"
        rag._ai.embed.return_value = [[0.1, 0.2]]
        rag._ai.expand_query.side_effect = Exception("no key")
        rag._ai._total_cost = 0.0
        rag._ai._total_input_tokens = 10
        rag._ai._total_output_tokens = 20
        rag._retriever = MagicMock()
        rag._retriever.query.return_value = [
            ChunkWithScore(text="context", score=0.9, source="doc.txt"),
        ]
        rag._mm_index = MagicMock()
        rag._mm_index.search.return_value = MagicMock(items=[])
        result = rag.query("test", cite=False, multi_query=False, hyde=False)
        assert hasattr(result, 'images')

    def test_multimodal_false_skips_mm_index(self):
        rag = AutoRAG(multimodal=False)
        rag._ai = MagicMock()
        assert rag._mm_index is None
        doc = Document(text="plain text")
        rag._chunk_and_index([doc])
        assert rag._mm_index is None


# ---------------------------------------------------------------------------
# Feature 3: Citation Engine + result.sources
# ---------------------------------------------------------------------------

class TestResultSources:
    def test_sources_property_returns_list(self):
        rag = AutoRAG()
        result = RAGResult(
            answer="Test answer.",
            citations=[
                Citation(text="source text", source="doc.pdf", page=1, relevance_score=0.9),
            ],
        )
        sources = result.sources
        assert len(sources) == 1
        assert sources[0]["source"] == "doc.pdf"
        assert sources[0]["page"] == 1
        assert sources[0]["score"] == 0.9

    def test_sources_empty_when_no_citations(self):
        result = RAGResult(answer="No citations.")
        assert result.sources == []

    def test_citation_engine_fallback_on_empty_citations(self):
        """When LLM returns answer without [N] markers, CitationEngine fallback fires."""
        rag = AutoRAG()
        mock_ai = MagicMock()
        mock_ai._call.return_value = "Revenue grew by 20% year over year."
        rag._ai = mock_ai
        chunks = [
            ChunkWithScore(text="Revenue grew by 20% year over year.", score=0.9, source="report.pdf"),
        ]
        answer, citations = rag._generate_answer(
            "How did revenue change?", chunks,
            return_citations=True, length="short",
        )
        assert answer == "Revenue grew by 20% year over year."


# ---------------------------------------------------------------------------
# Feature 5: Retriever Circuit Breaker
# ---------------------------------------------------------------------------

class TestRetrieverCircuitBreaker:
    def test_circuit_breaker_opens_after_consecutive_failures(self):
        rag = AutoRAG()
        rag._ai = MagicMock()
        rag._ai.embed.return_value = [[0.1, 0.2]]
        rag._retriever_cb_threshold = 2
        rag._retriever = MagicMock()
        rag._retriever.query.side_effect = Exception("DB down")
        result1 = rag._retrieve("test")
        assert rag._retriever_failures == 1
        assert not rag._retriever_cb_open
        result2 = rag._retrieve("test")
        assert rag._retriever_failures == 2
        assert rag._retriever_cb_open
        result3 = rag._retrieve("test")
        assert len(result3) == 0  # CB open returns empty

    def test_circuit_breaker_resets_on_success(self):
        rag = AutoRAG()
        rag._ai = MagicMock()
        rag._ai.embed.return_value = [[0.1, 0.2]]
        rag._retriever_failures = 2
        rag._retriever = MagicMock()
        rag._retriever.query.return_value = [ChunkWithScore(text="ok", score=0.9, source="doc.txt")]
        result = rag._retrieve("test")
        assert rag._retriever_failures == 0
        assert len(result) == 1

    def test_robust_rag_fallback_on_cb_open(self):
        """Circuit breaker triggers early return, RobustRAG handles graceful degradation."""
        rag = AutoRAG()
        rag._ai = MagicMock()
        rag._retriever_cb_open = True
        result = rag._retrieve("test")
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Feature wiring tests — auto_rag() / AutoRAG parameter wiring
# ---------------------------------------------------------------------------

class TestAutoRAGFeatureWiring:
    """Verify that Tier 2 feature parameters correctly initialize sub-components."""

    def test_multi_language_wiring(self):
        rag = AutoRAG(multi_language=True, languages=["en", "es"])
        assert rag.multi_language is True
        assert rag._languages == ["en", "es"]
        assert rag._multilingual is not None
        assert rag._multilingual.languages == ["en", "es"]

    def test_multi_language_false_by_default(self):
        rag = AutoRAG()
        assert rag.multi_language is False
        assert rag._multilingual is None

    def test_reasoning_wiring(self):
        rag = AutoRAG(reasoning=True, reasoning_max_steps=3)
        assert rag.reasoning_enabled is True
        assert rag._reasoning_max_steps == 3
        assert rag._reasoner is not None
        assert rag._reasoner.max_steps == 3

    def test_reasoning_false_by_default(self):
        rag = AutoRAG()
        assert rag.reasoning_enabled is False
        assert rag._reasoner is None

    def test_smart_routing_wiring(self):
        child = AutoRAG()
        rag = AutoRAG(routing_rags={"tech": child})
        assert rag._router_v2 is not None
        assert "tech" in rag._routing_rags

    def test_smart_routing_false_by_default(self):
        rag = AutoRAG()
        assert rag._router_v2 is None
        assert rag._routing_rags is None

    def test_experiment_wiring(self):
        config = {"name": "test_exp", "variants": {"control": {"top_k": 3}}}
        rag = AutoRAG(experiment_config=config)
        assert rag._experiment_manager is not None
        assert rag._experiment_manager.name == "test_exp"

    def test_experiment_false_by_default(self):
        rag = AutoRAG()
        assert rag._experiment_manager is None
        assert rag._experiment_config is None

    def test_auto_rag_factory_passes_tier2_params(self):
        from runeextract.rag.auto_pipeline import auto_rag
        with patch("runeextract.rag.auto_pipeline.extract") as mock_extract:
            mock_extract.return_value = Document(
                text="test", source_type="text", source_path="t.txt",
            )
            rag = auto_rag("t.txt", multi_language=True)
            assert rag.multi_language is True
            assert rag._multilingual is not None

    def test_all_tier2_params_off_by_default(self):
        rag = AutoRAG()
        assert rag._multilingual is None
        assert rag._reasoner is None
        assert rag._router_v2 is None
        assert rag._experiment_manager is None


class TestAutoRAGQueryFeatureDispatch:
    """Verify that query() correctly dispatches to feature wrappers."""

    def test_query_routes_to_multilingual(self):
        rag = AutoRAG(multi_language=True)
        mock_ml = MagicMock()
        mock_ml.query.return_value = RAGResult(
            answer="translated", citations=[], confidence=0.9,
            retrieved_chunks=[], query_variants=[], latency_ms=0,
            tokens_used={}, cost=0.0, total_session_cost=0.0
        )
        rag._multilingual = mock_ml
        result = rag.query("¿Hola?")
        mock_ml.query.assert_called_once()
        assert result.answer == "translated"

    def test_query_routes_to_reasoner(self):
        rag = AutoRAG(reasoning=True)
        mock_r = MagicMock()
        mock_r.reason.return_value = RAGResult(
            answer="reasoned", citations=[], confidence=0.9,
            retrieved_chunks=[], query_variants=[], latency_ms=0,
            tokens_used={}, cost=0.0, total_session_cost=0.0
        )
        rag._reasoner = mock_r
        result = rag.query("Why is the sky blue?", reasoning=True)
        mock_r.reason.assert_called_once()
        assert result.answer == "reasoned"

    def test_query_routes_to_router(self):
        child = AutoRAG()
        rag = AutoRAG(routing_rags={"default": child})
        mock_router = MagicMock()
        mock_router.query.return_value = RAGResult(
            answer="routed", citations=[], confidence=0.9,
            retrieved_chunks=[], query_variants=[], latency_ms=0,
            tokens_used={}, cost=0.0, total_session_cost=0.0
        )
        rag._router_v2 = mock_router
        result = rag.query("How does routing work?")
        mock_router.query.assert_called_once()
        assert result.answer == "routed"

    def test_query_routes_to_experiment_manager(self):
        config = {"name": "exp", "variants": {"ctrl": {"top_k": 3}}}
        rag = AutoRAG(experiment_config=config)
        mock_exp = MagicMock()
        mock_exp.query.return_value = RAGResult(
            answer="experiment", citations=[], confidence=0.9,
            retrieved_chunks=[], query_variants=[], latency_ms=0,
            tokens_used={}, cost=0.0, total_session_cost=0.0
        )
        rag._experiment_manager = mock_exp
        result = rag.query("Test?", user_id="user1")
        mock_exp.query.assert_called_once()
        assert result.answer == "experiment"


class TestAutoRAGRBACQuery:
    """Verify RBAC user/roles params in query()."""

    def test_query_passes_user_to_rbac(self):
        rag = AutoRAG(rbac=True)
        rag._ai = MagicMock()
        rag._ai.embed.return_value = [[0.1] * 384]
        rag._ai._call.return_value = "Answer. [1]"
        rag._ai._total_input_tokens = 0
        rag._ai._total_output_tokens = 0
        rag._retriever = MagicMock()
        rag._retriever.query.return_value = [
            ChunkWithScore(text="content", score=0.9, source="doc.txt"),
        ]
        mock_rbac = MagicMock()
        mock_rbac.filter_chunks.return_value = [
            ChunkWithScore(text="content", score=0.9, source="doc.txt"),
        ]
        rag._rbac = mock_rbac
        result = rag.query("Question?", user="alice", roles=["admin"])
        mock_rbac.filter_chunks.assert_called_once()
        args, kwargs = mock_rbac.filter_chunks.call_args
        assert kwargs["user"] == "alice"
        assert kwargs["roles"] == ["admin"]

    def test_query_rbac_defaults(self):
        rag = AutoRAG(rbac=True)
        rag._ai = MagicMock()
        rag._ai.embed.return_value = [[0.1] * 384]
        rag._ai._call.return_value = "Answer. [1]"
        rag._ai._total_input_tokens = 0
        rag._ai._total_output_tokens = 0
        rag._retriever = MagicMock()
        rag._retriever.query.return_value = [
            ChunkWithScore(text="content", score=0.9, source="doc.txt"),
        ]
        mock_rbac = MagicMock()
        mock_rbac.filter_chunks.return_value = [
            ChunkWithScore(text="content", score=0.9, source="doc.txt"),
        ]
        rag._rbac = mock_rbac
        result = rag.query("Question?")
        mock_rbac.filter_chunks.assert_called_once()
        args, kwargs = mock_rbac.filter_chunks.call_args
        assert kwargs["user"] == "anonymous"
        assert kwargs["roles"] == []


class TestAutoRAGAPIServer:
    """Verify create_api_server() and serve() methods."""

    def test_create_api_server(self):
        rag = AutoRAG()
        api = rag.create_api_server(api_keys=["key1"], rate_limit=50)
        assert api is not None
        assert api.rag is rag
        assert api.api_keys == {"key1"}
        assert api.rate_limit == 50

    def test_create_api_server_default_rate_limit(self):
        rag = AutoRAG()
        api = rag.create_api_server()
        assert api.rate_limit == 100
        assert api.api_keys is None

    def test_serve_logs_error_when_uvicorn_missing(self):
        rag = AutoRAG()
        with patch.dict("sys.modules", uvicorn=None):
            with patch("runeextract.rag.auto_pipeline.logger") as mock_log:
                rag.serve(host="127.0.0.1", port=9999)
                assert mock_log.error.called
