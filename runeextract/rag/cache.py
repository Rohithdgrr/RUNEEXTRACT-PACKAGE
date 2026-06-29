"""
Multi-level caching for RAG pipelines.

Provides an LRU cache with time-to-live (TTL) expiry for embeddings,
vector-search results, and generated answers.
"""

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

from runeextract.rag.types import ChunkWithScore

logger = logging.getLogger(__name__)


class _LRUTTLDict:
    """Thread-safe LRU dict with per-item TTL expiry."""

    def __init__(self, maxsize: int = 1000, default_ttl: float = 300.0):
        self._maxsize = maxsize
        self._default_ttl = default_ttl
        self._data: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._data:
                return None
            expires, value = self._data[key]
            if time.monotonic() > expires:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        expires = time.monotonic() + (ttl if ttl is not None else self._default_ttl)
        with self._lock:
            self._data[key] = (expires, value)
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def invalidate(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


def _hash(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


class RAGCache:
    """Multi-level RAG cache with automatic TTL and LRU eviction.

    Three cache levels:

    * ``embedding_cache`` — ``{text_hash → embedding_vector}`` (TTL 3600s)
    * ``search_cache`` — ``{(query_hash, top_k, metadata_filter) → chunks}`` (TTL 300s)
    * ``answer_cache`` — ``{(question_hash) → answer}`` (TTL 600s)
    """

    def __init__(self, maxsize: int = 1000):
        self.embedding_cache = _LRUTTLDict(maxsize=maxsize, default_ttl=3600.0)
        self.search_cache = _LRUTTLDict(maxsize=maxsize // 2, default_ttl=300.0)
        self.answer_cache = _LRUTTLDict(maxsize=maxsize // 5, default_ttl=600.0)

    # -- embedding cache ---------------------------------------------------

    def get_embedding(self, text: str) -> Optional[List[float]]:
        key = _hash(text)
        return self.embedding_cache.get(key)

    def put_embedding(self, text: str, embedding: List[float]) -> None:
        key = _hash(text)
        self.embedding_cache.put(key, embedding)

    # -- search cache ------------------------------------------------------

    def _search_key(self, query: str, top_k: int,
                    metadata_filter: Optional[Dict[str, Any]] = None) -> str:
        filter_str = "" if metadata_filter is None else str(sorted(metadata_filter.items()))
        return _hash(query, str(top_k), filter_str)

    def get_search(self, query: str, top_k: int,
                   metadata_filter: Optional[Dict[str, Any]] = None) -> Optional[List[ChunkWithScore]]:
        return self.search_cache.get(self._search_key(query, top_k, metadata_filter))

    def put_search(self, query: str, top_k: int, chunks: List[ChunkWithScore],
                   metadata_filter: Optional[Dict[str, Any]] = None) -> None:
        self.search_cache.put(self._search_key(query, top_k, metadata_filter), chunks)

    # -- answer cache ------------------------------------------------------

    def get_answer(self, question: str) -> Optional[str]:
        return self.answer_cache.get(_hash(question))

    def put_answer(self, question: str, answer: str) -> None:
        self.answer_cache.put(_hash(question), answer)

    # -- lifecycle ---------------------------------------------------------

    def invalidate(self) -> None:
        """Clear all cache levels."""
        self.embedding_cache.invalidate()
        self.search_cache.invalidate()
        self.answer_cache.invalidate()
        logger.debug("RAGCache invalidated")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.invalidate()
