"""
Tests for RAG Phase-1 Foundation: instant_rag, templates, cache, embedding_selector.
"""

from unittest.mock import Mock, patch

import pytest

from runeextract.rag.cache import RAGCache
from runeextract.rag.embedding_selector import resolve_embedding, get_domain_embedding
from runeextract.rag.templates import DomainConfig, DomainTemplates
from runeextract.rag.types import ChunkWithScore


# ---------------------------------------------------------------------------
# DomainTemplates
# ---------------------------------------------------------------------------

class TestDomainTemplates:
    def setup_method(self):
        DomainTemplates.reset()
    def test_get_known_domain(self):
        config = DomainTemplates.get("financial")
        assert config.chunking == "by_heading"
        assert config.chunk_size == 800
        assert config.reranker == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert "financial" in config.system_prompt.lower()

    def test_get_legal(self):
        config = DomainTemplates.get("legal")
        assert config.chunk_size == 1200
        assert config.chunk_overlap == 100

    def test_get_medical(self):
        config = DomainTemplates.get("medical")
        assert config.chunking == "sentence_window"
        assert config.chunk_size == 600

    def test_get_academic(self):
        config = DomainTemplates.get("academic")
        assert config.chunking == "hierarchical"
        assert config.chunk_size == 1000

    def test_get_unknown_domain_returns_default(self):
        config = DomainTemplates.get("nonexistent")
        assert config.chunking == "auto"
        assert config.chunk_size == 1000

    def test_list_contains_all_defaults(self):
        templates = DomainTemplates.list()
        assert set(templates) == {"financial", "legal", "medical", "academic"}

    def test_register_custom(self):
        custom = DomainConfig(chunking="fixed_size", chunk_size=500)
        DomainTemplates.register("custom", custom)
        retrieved = DomainTemplates.get("custom")
        assert retrieved.chunking == "fixed_size"
        assert retrieved.chunk_size == 500

    def test_register_overrides_existing(self):
        DomainTemplates.register("financial", DomainConfig(chunking="fixed_size"))
        retrieved = DomainTemplates.get("financial")
        assert retrieved.chunking == "fixed_size"

    def test_domain_config_defaults(self):
        c = DomainConfig()
        assert c.chunking == "auto"
        assert c.chunk_size == 1000
        assert c.chunk_overlap == 100
        assert c.reranker is None
        assert c.embedding == "auto"


# ---------------------------------------------------------------------------
# EmbeddingSelector
# ---------------------------------------------------------------------------

class TestEmbeddingSelector:
    def setup_method(self):
        DomainTemplates.reset()
    def test_resolve_fast(self):
        assert resolve_embedding("fast") == "openai:text-embedding-3-small"

    def test_resolve_balanced(self):
        assert resolve_embedding("balanced") == "openai:text-embedding-3-large"

    def test_resolve_accurate(self):
        assert resolve_embedding("accurate") == "openai:text-embedding-3-large"

    def test_resolve_passthrough(self):
        spec = "custom:my-model-v2"
        assert resolve_embedding(spec) == spec

    def test_domain_embedding_medical(self):
        emb = get_domain_embedding("medical")
        assert emb == "openai:text-embedding-3-large"

    def test_domain_embedding_financial(self):
        emb = get_domain_embedding("financial")
        assert emb == "openai:text-embedding-3-large"

    def test_domain_embedding_unknown(self):
        emb = get_domain_embedding("nonexistent")
        assert emb == "openai:text-embedding-3-large"


# ---------------------------------------------------------------------------
# RAGCache
# ---------------------------------------------------------------------------

class TestRAGCache:
    def test_embedding_hit(self):
        cache = RAGCache()
        assert cache.get_embedding("hello") is None
        cache.put_embedding("hello", [0.1, 0.2, 0.3])
        assert cache.get_embedding("hello") == [0.1, 0.2, 0.3]

    def test_embedding_miss(self):
        cache = RAGCache()
        assert cache.get_embedding("world") is None

    def test_search_hit(self):
        cache = RAGCache()
        chunks = [ChunkWithScore(text="test", score=0.9)]
        assert cache.get_search("query", 5) is None
        cache.put_search("query", 5, chunks)
        result = cache.get_search("query", 5)
        assert result is not None
        assert result[0].text == "test"

    def test_search_miss_different_top_k(self):
        cache = RAGCache()
        chunks = [ChunkWithScore(text="test", score=0.9)]
        cache.put_search("query", 5, chunks)
        assert cache.get_search("query", 10) is None

    def test_answer_hit(self):
        cache = RAGCache()
        assert cache.get_answer("question?") is None
        cache.put_answer("question?", "answer text")
        assert cache.get_answer("question?") == "answer text"

    def test_invalidate_all(self):
        cache = RAGCache()
        cache.put_embedding("a", [1.0])
        cache.put_search("q", 5, [ChunkWithScore(text="t", score=0.5)])
        cache.put_answer("q?", "a")
        cache.invalidate()
        assert cache.get_embedding("a") is None
        assert cache.get_search("q", 5) is None
        assert cache.get_answer("q?") is None

    def test_context_manager(self):
        with RAGCache() as cache:
            cache.put_embedding("x", [0.5])
            assert cache.get_embedding("x") == [0.5]
        assert cache.get_embedding("x") is None


# ---------------------------------------------------------------------------
# instant_rag factory
# ---------------------------------------------------------------------------

class TestInstantRAG:
    def setup_method(self):
        DomainTemplates.reset()
    def test_importable(self):
        from runeextract.rag.instant import instant_rag
        assert callable(instant_rag)

    def test_top_level_importable(self):
        from runeextract import instant_rag as ir
        assert callable(ir)

    def test_returns_auto_rag(self):
        from runeextract.rag.instant import instant_rag
        with patch("runeextract.rag.instant._auto_rag") as mock_factory:
            mock_factory.return_value = "pipeline"
            result = instant_rag("test.txt", model="openai:gpt-4o-mini")
            assert result == "pipeline"

    def test_passes_default_kwargs(self):
        from runeextract.rag.instant import instant_rag
        with patch("runeextract.rag.instant._auto_rag") as mock_factory:
            mock_factory.return_value = "pipeline"
            instant_rag("test.txt")
            call_kwargs = mock_factory.call_args[1]
            assert call_kwargs["vector_store"] == "faiss"
            assert call_kwargs["llm"] == "openai:gpt-4o-mini"

    def test_domain_applies_config(self):
        from runeextract.rag.instant import instant_rag
        with patch("runeextract.rag.instant._auto_rag") as mock_factory:
            mock_factory.return_value = "pipeline"
            instant_rag("test.txt", domain="financial")
            call_kwargs = mock_factory.call_args[1]
            assert call_kwargs["chunking"] == "by_heading"
            assert call_kwargs["chunk_size"] == 800

    def test_custom_kwargs_override_domain(self):
        from runeextract.rag.instant import instant_rag
        with patch("runeextract.rag.instant._auto_rag") as mock_factory:
            mock_factory.return_value = "pipeline"
            instant_rag("test.txt", domain="financial", chunk_size=999)
            call_kwargs = mock_factory.call_args[1]
            assert call_kwargs["chunk_size"] == 999
