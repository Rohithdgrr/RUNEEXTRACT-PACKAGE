"""
Optional caching layer using diskcache.

Cache keys are based on a hash of (file_path, file_mtime, options).
Supports zlib compression for the JSON fallback backend and
enhanced TTL/staleness checks with max cache size enforcement.
"""

import gzip
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable

from runeextract.models.document import Document

logger = logging.getLogger(__name__)

# Fields to exclude from cache to avoid persisting sensitive data
_SENSITIVE_OPTION_KEYS = {"password", "api_key", "token", "secret", "credential", "auth"}


class ExtractionCache:
    """
    Disk-backed cache for extraction results.

    Uses diskcache if available, falls back to compressed JSON file cache.
    Supports configurable TTL, optional compression, and max cache size.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        ttl: int = 3600,
        compress: bool = True,
        max_size_mb: float = 0.0,
    ):
        safe_dir = cache_dir or str(Path.home() / ".runeextract_cache")
        self.cache_dir = Path(safe_dir)
        self.ttl = ttl
        self.compress = compress
        self.max_size_bytes = int(max_size_mb * 1024 * 1024) if max_size_mb > 0 else 0
        self._diskcache = None
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._init_backend()

    @property
    def hits(self) -> int:
        """Number of cache hits since creation."""
        return self._hits

    @property
    def misses(self) -> int:
        """Number of cache misses since creation."""
        return self._misses

    @property
    def evictions(self) -> int:
        """Number of cache evictions since creation."""
        return self._evictions

    def reset_stats(self):
        """Reset hit/miss/eviction counters."""
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _init_backend(self):
        try:
            import diskcache as dc
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._diskcache = dc.Cache(str(self.cache_dir))
            logger.debug(f"Caching with diskcache at {self.cache_dir}")
        except ImportError:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Caching with compressed JSON at {self.cache_dir} (diskcache not installed)")

    @staticmethod
    def _strip_sensitive_options(options: dict) -> dict:
        """Remove sensitive keys from options before cache key generation."""
        return {k: v for k, v in options.items()
                if k.lower() not in _SENSITIVE_OPTION_KEYS
                and not any(s in k.lower() for s in _SENSITIVE_OPTION_KEYS)}

    def _make_key(self, file_path: str, options: dict) -> str:
        safe_opts = self._strip_sensitive_options(options)
        try:
            mtime = os.path.getmtime(file_path)
        except OSError:
            mtime = 0
        raw = json.dumps({"path": file_path, "mtime": mtime, "opts": safe_opts}, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_file_path(self, key: str) -> Path:
        suffix = ".gz" if self.compress else ".json"
        return self.cache_dir / f"{key}{suffix}"

    def get(self, file_path: str, options: dict) -> Optional[Document]:
        key = self._make_key(file_path, options)
        if self._diskcache:
            data = self._diskcache.get(key)
        else:
            cache_file = self._cache_file_path(key)
            if not cache_file.exists():
                self._misses += 1
                return None
            try:
                if self.compress:
                    raw = gzip.decompress(cache_file.read_bytes())
                else:
                    raw = cache_file.read_bytes()
                data = json.loads(raw.decode("utf-8"))
            except (OSError, json.JSONDecodeError, gzip.BadGzipFile) as exc:
                logger.debug(f"Cache read error: {exc}")
                self._misses += 1
                return None

        if data is None:
            self._misses += 1
            return None

        cached_at = data.get("_cached_at", 0)
        if time.time() - cached_at > self.ttl:
            try:
                if self._diskcache:
                    del self._diskcache[key]
                else:
                    self._cache_file_path(key).unlink(missing_ok=True)
            except (KeyError, OSError) as exc:
                logger.debug(f"Cache TTL cleanup error: {exc}")
            self._misses += 1
            return None

        try:
            # Restore full Document including tables and images
            doc = Document(
                text=data.get("text", ""),
                source_type=data.get("source_type", ""),
                source_path=data.get("source_path"),
                metadata=data.get("metadata", {}),
            )
            # Restore tables
            tables_data = data.get("tables")
            if tables_data:
                try:
                    from runeextract.extractors.base import ExtractedTable
                    doc._tables = [ExtractedTable(**t) if isinstance(t, dict) else t for t in tables_data]
                except Exception:
                    pass
            # Restore image refs (not raw bytes to avoid memory bloat)
            images_data = data.get("images")
            if images_data:
                doc._images = images_data
            self._hits += 1
            return doc
        except (TypeError, ValueError) as exc:
            logger.debug(f"Cache deserialize error: {exc}")
            self._misses += 1
            return None

    def set(self, file_path: str, options: dict, document: Document) -> None:
        # Strip sensitive options before using as cache key
        safe_options = self._strip_sensitive_options(options)
        key = self._make_key(file_path, safe_options)
        data = document.to_dict()
        # Strip passwords and secrets from cached data
        data.pop("password", None)
        data["_cached_at"] = time.time()
        data["_file_path"] = file_path
        if self._diskcache:
            self._diskcache.set(key, data, expire=self.ttl)
        else:
            self._enforce_max_size()
            cache_file = self._cache_file_path(key)
            try:
                payload = json.dumps(data, default=str).encode("utf-8")
                if self.compress:
                    cache_file.write_bytes(gzip.compress(payload))
                else:
                    cache_file.write_bytes(payload)
            except (OSError, TypeError) as exc:
                logger.debug(f"Cache write error: {exc}")

    def invalidate(self, file_path: str) -> None:
        if self._diskcache:
            for k in list(self._diskcache.iterkeys()):
                try:
                    entry = self._diskcache.get(k)
                    if isinstance(entry, dict) and entry.get("_file_path") == file_path:
                        del self._diskcache[k]
                except (KeyError, OSError) as exc:
                    logger.debug(f"Cache invalidate error: {exc}")
        else:
            suffix = ".gz" if self.compress else ".json"
            for f in self.cache_dir.glob(f"*{suffix}"):
                try:
                    raw = gzip.decompress(f.read_bytes()) if self.compress else f.read_bytes()
                    data = json.loads(raw.decode("utf-8"))
                    if data.get("_file_path") == file_path:
                        f.unlink()
                except (OSError, json.JSONDecodeError, gzip.BadGzipFile) as exc:
                    logger.debug(f"Cache invalidate error: {exc}")

    def close(self) -> None:
        if self._diskcache:
            self._diskcache.close()

    def _enforce_max_size(self):
        """Evict oldest entries when cache exceeds max_size_bytes."""
        if self.max_size_bytes <= 0:
            return
        suffix = "*" + (".gz" if self.compress else ".json")
        files = sorted(self.cache_dir.glob(suffix), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        while total > self.max_size_bytes and len(files) > 1:
            oldest = files.pop(0)
            total -= oldest.stat().st_size
            try:
                oldest.unlink()
                self._evictions += 1
            except OSError:
                pass
