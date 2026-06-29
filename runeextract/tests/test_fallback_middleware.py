"""Tests for provider fallback chain and RAG middleware pipeline."""

import pytest
from unittest.mock import MagicMock, patch


# ---- Provider Fallback Chain ----

class TestProviderFallback:
    def test_fallback_providers_init(self):
        from runeextract.processors.ai import AIProcessor
        fb = [{"provider": "anthropic", "api_key": "sk-test"}]
        ai = AIProcessor(api_key="sk-test", provider="openai", fallback_providers=fb)
        assert ai.fallback_providers == fb
        assert ai.provider == "openai"

    def test_get_provider_chain_primary_only(self):
        from runeextract.processors.ai import AIProcessor
        ai = AIProcessor(api_key="sk-test", provider="openai")
        chain = ai._get_provider_chain()
        assert len(chain) == 1
        assert chain[0][0] == "openai"

    def test_get_provider_chain_with_fallbacks(self):
        from runeextract.processors.ai import AIProcessor
        fb = [{"provider": "anthropic", "api_key": "ak-test"}]
        ai = AIProcessor(api_key="sk-test", provider="openai", fallback_providers=fb)
        chain = ai._get_provider_chain()
        assert len(chain) == 2
        assert chain[0][0] == "openai"
        assert chain[1][0] == "anthropic"

    def test_get_provider_chain_deduplicates(self):
        from runeextract.processors.ai import AIProcessor
        fb = [{"provider": "openai", "api_key": "sk-test2"}]
        ai = AIProcessor(api_key="sk-test", provider="openai", fallback_providers=fb)
        chain = ai._get_provider_chain()
        assert len(chain) == 1

    def test_shared_thread_pool(self):
        from runeextract.processors.ai import AIProcessor
        ai = AIProcessor(api_key="sk-test", provider="openai")
        pool1 = ai._get_thread_pool(4)
        pool2 = ai._get_thread_pool(4)
        assert pool1 is pool2

    def test_close_shuts_down_thread_pool(self):
        from runeextract.processors.ai import AIProcessor
        ai = AIProcessor(api_key="sk-test", provider="openai")
        pool = ai._get_thread_pool(4)
        ai.close()
        assert pool._shutdown


# ---- RAG Middleware Pipeline ----

class TestMiddlewarePipeline:
    def test_pipeline_empty(self):
        from runeextract.rag.middleware import MiddlewarePipeline, RAGMiddleware
        pipeline = MiddlewarePipeline()
        assert len(pipeline._middlewares) == 0

    def test_pipeline_add_remove(self):
        from runeextract.rag.middleware import MiddlewarePipeline, LoggingMiddleware
        pipeline = MiddlewarePipeline()
        pipeline.add(LoggingMiddleware())
        assert len(pipeline._middlewares) == 1
        pipeline.remove(LoggingMiddleware)
        assert len(pipeline._middlewares) == 0

    def test_wrap_basic(self):
        from runeextract.rag.middleware import MiddlewarePipeline
        pipeline = MiddlewarePipeline()
        fn = MagicMock(return_value="result")
        wrapped = pipeline.wrap(fn)
        assert wrapped() == "result"
        fn.assert_called_once()

    def test_wrap_with_args(self):
        from runeextract.rag.middleware import MiddlewarePipeline
        pipeline = MiddlewarePipeline()
        fn = MagicMock(return_value="hello")
        wrapped = pipeline.wrap(fn)
        assert wrapped("arg1", key="val") == "hello"
        fn.assert_called_with("arg1", key="val")

    def test_wrap_preserves_name(self):
        from runeextract.rag.middleware import MiddlewarePipeline
        pipeline = MiddlewarePipeline()
        def my_query(): pass
        wrapped = pipeline.wrap(my_query)
        assert wrapped.__name__ == "my_query"


class TestCacheMiddleware:
    def test_cache_hit(self):
        from runeextract.rag.middleware import CacheMiddleware
        mw = CacheMiddleware(maxsize=10)
        context = {"args": ("hello",), "kwargs": {}}
        fn = MagicMock(return_value="result")
        result1 = mw.process(context, fn)
        result2 = mw.process(context, fn)
        assert result1 == result2 == "result"
        assert fn.call_count == 1
        assert mw.hits == 1
        assert mw.misses == 1

    def test_cache_maxsize(self):
        from runeextract.rag.middleware import CacheMiddleware
        mw = CacheMiddleware(maxsize=2)
        fn = MagicMock(return_value="val")
        for i in range(3):
            mw.process({"args": (f"key{i}",), "kwargs": {}}, fn)
        assert fn.call_count == 3
        assert mw.misses == 3

    def test_cache_different_args(self):
        from runeextract.rag.middleware import CacheMiddleware
        mw = CacheMiddleware()
        fn = MagicMock(side_effect=["a", "b"])
        r1 = mw.process({"args": ("x",), "kwargs": {}}, fn)
        r2 = mw.process({"args": ("y",), "kwargs": {}}, fn)
        assert r1 == "a"
        assert r2 == "b"
        assert fn.call_count == 2


class TestFallbackMiddleware:
    def test_fallback_on_success(self):
        from runeextract.rag.middleware import FallbackMiddleware
        mw = FallbackMiddleware()
        fn = MagicMock(return_value="success")
        result = mw.process({"args": (), "kwargs": {}}, fn)
        assert result == "success"

    def test_fallback_on_failure(self):
        from runeextract.rag.middleware import FallbackMiddleware
        from runeextract.rag.types import RAGResult
        mw = FallbackMiddleware(fallback_text="sorry")
        fn = MagicMock(side_effect=RuntimeError("fail"))
        result = mw.process({"args": (), "kwargs": {}}, fn)
        assert result.answer == "sorry"


class TestLoggingMiddleware:
    def test_logging_passes_through(self):
        from runeextract.rag.middleware import LoggingMiddleware
        mw = LoggingMiddleware()
        fn = MagicMock(return_value="result")
        result = mw.process({"args": (), "kwargs": {}}, fn)
        assert result == "result"

    def test_logging_raises_on_error(self):
        from runeextract.rag.middleware import LoggingMiddleware
        mw = LoggingMiddleware()
        fn = MagicMock(side_effect=ValueError("bad"))
        with pytest.raises(ValueError):
            mw.process({"args": (), "kwargs": {}}, fn)


class TestTimingMiddleware:
    def test_timing_tracked(self):
        from runeextract.rag.middleware import TimingMiddleware
        import time
        mw = TimingMiddleware()
        fn = MagicMock(return_value="result")
        mw.process({"args": (), "kwargs": {}}, fn)
        assert mw.query_count == 1
        assert mw.last_duration >= 0
        assert mw.total_duration >= 0


class TestApplyMiddleware:
    def test_apply_middleware(self):
        from runeextract.rag.middleware import apply_middleware, LoggingMiddleware

        class FakeRAG:
            def query(self, question: str = ""):
                return f"answer to {question}"

        rag = FakeRAG()
        apply_middleware(rag, [LoggingMiddleware()])
        assert rag.query("hello") == "answer to hello"
