"""RAG middleware pipeline — composable middleware for query processing.

Enables decorating an AutoRAG instance with cross-cutting concerns
(caching, routing, RBAC, logging, etc.) without modifying the core class.
"""

import functools
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class RAGMiddleware:
    """Base middleware class. Subclasses implement process()."""

    def process(self, context: Dict[str, Any], next_fn: Callable) -> Any:
        return next_fn(context)


class MiddlewarePipeline:
    """Chain of middlewares wrapping a RAG query function."""

    def __init__(self, middlewares: Optional[List[RAGMiddleware]] = None):
        self._middlewares: List[RAGMiddleware] = list(middlewares or [])

    def add(self, middleware: RAGMiddleware):
        self._middlewares.append(middleware)

    def remove(self, middleware_type: Type[RAGMiddleware]):
        self._middlewares = [m for m in self._middlewares if not isinstance(m, middleware_type)]

    def wrap(self, query_fn: Callable) -> Callable:
        @functools.wraps(query_fn)
        def wrapped(*args, **kwargs):
            context = {"args": args, "kwargs": kwargs, "result": None}
            chain = list(self._middlewares)

            def execute(ctx):
                if chain:
                    mw = chain.pop(0)
                    return mw.process(ctx, execute)
                ctx["result"] = query_fn(*ctx["args"], **ctx["kwargs"])
                return ctx["result"]

            return execute(context)
        return wrapped


class LoggingMiddleware(RAGMiddleware):
    """Log query timing and basic info."""

    def process(self, context: Dict[str, Any], next_fn: Callable) -> Any:
        start = time.time()
        try:
            result = next_fn(context)
            elapsed = time.time() - start
            logger.info("RAG query completed in %.2fs", elapsed)
            return result
        except Exception as exc:
            elapsed = time.time() - start
            logger.error("RAG query failed after %.2fs: %s", elapsed, exc)
            raise


class TimingMiddleware(RAGMiddleware):
    """Capture timing stats per query."""

    def __init__(self):
        self.last_duration = 0.0
        self.total_duration = 0.0
        self.query_count = 0

    def process(self, context: Dict[str, Any], next_fn: Callable) -> Any:
        start = time.time()
        try:
            result = next_fn(context)
            return result
        finally:
            self.last_duration = time.time() - start
            self.total_duration += self.last_duration
            self.query_count += 1


class CacheMiddleware(RAGMiddleware):
    """Wrap a RAG query with an LRU cache."""

    def __init__(self, maxsize: int = 128):
        self._cache: Dict[str, Any] = {}
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def _make_key(self, context: Dict[str, Any]) -> str:
        kwargs = context.get("kwargs", {})
        args = context.get("args", ())
        import hashlib, json
        raw = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()

    def process(self, context: Dict[str, Any], next_fn: Callable) -> Any:
        key = self._make_key(context)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        result = next_fn(context)
        if len(self._cache) >= self.maxsize:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = result
        return result


class FallbackMiddleware(RAGMiddleware):
    """Return a fallback response on query failure."""

    def __init__(self, fallback_text: str = "I'm sorry, I couldn't process that query."):
        self.fallback_text = fallback_text

    def process(self, context: Dict[str, Any], next_fn: Callable) -> Any:
        try:
            return next_fn(context)
        except Exception as exc:
            logger.warning("Query failed, returning fallback: %s", exc)
            from runeextract.rag.types import RAGResult
            return RAGResult(
                answer=self.fallback_text,
                citations=[],
            )


def apply_middleware(rag_instance: Any, middlewares: Optional[List[RAGMiddleware]] = None) -> Any:
    """Wrap a RAG instance's query() method with middleware.

    Usage:
        rag = auto_rag("./docs")
        apply_middleware(rag, [LoggingMiddleware(), CacheMiddleware()])
        rag.query("What is AI?")  # Now runs through middleware
    """
    pipeline = MiddlewarePipeline(middlewares)
    rag_instance.query = pipeline.wrap(rag_instance.query.__get__(rag_instance, type(rag_instance)))
    return rag_instance
