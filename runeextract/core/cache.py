"""
Optional caching layer using diskcache.

Cache keys are based on a hash of (file_path, file_mtime, options).
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable

from runeextract.models.document import Document

logger = logging.getLogger(__name__)


class ExtractionCache:
    """
    Disk-backed cache for extraction results.

    Uses diskcache if available, falls back to JSON file cache.
    """

    def __init__(self, cache_dir: Optional[str] = None, ttl: int = 3600):
        self.cache_dir = Path(cache_dir or (Path.home() / ".runeextract_cache"))
        self.ttl = ttl
        self._diskcache = None
        self._init_backend()

    def _init_backend(self):
        try:
            import diskcache as dc
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._diskcache = dc.Cache(str(self.cache_dir))
            logger.debug(f"Caching with diskcache at {self.cache_dir}")
        except ImportError:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Caching with JSON files at {self.cache_dir} (diskcache not installed)")

    def _make_key(self, file_path: str, options: dict) -> str:
        try:
            mtime = os.path.getmtime(file_path)
        except OSError:
            mtime = 0
        raw = json.dumps({"path": file_path, "mtime": mtime, "opts": options}, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, file_path: str, options: dict) -> Optional[Document]:
        key = self._make_key(file_path, options)
        if self._diskcache:
            data = self._diskcache.get(key)
        else:
            cache_file = self.cache_dir / f"{key}.json"
            if not cache_file.exists():
                return None
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
            except Exception:
                return None

        if data is None:
            return None

        if not self._diskcache:
            cached_at = data.get("_cached_at", 0)
            if time.time() - cached_at > self.ttl:
                try:
                    (self.cache_dir / f"{key}.json").unlink()
                except OSError:
                    pass
                return None

        try:
            doc = Document(
                text=data.get("text", ""),
                source_type=data.get("source_type", ""),
                source_path=data.get("source_path"),
                metadata=data.get("metadata", {}),
            )
            return doc
        except Exception as exc:
            logger.debug(f"Cache deserialize error: {exc}")
            return None

    def set(self, file_path: str, options: dict, document: Document) -> None:
        key = self._make_key(file_path, options)
        data = document.to_dict()
        data["_cached_at"] = time.time()
        data["_file_path"] = file_path
        if self._diskcache:
            self._diskcache.set(key, data, expire=self.ttl)
        else:
            cache_file = self.cache_dir / f"{key}.json"
            try:
                cache_file.write_text(json.dumps(data, default=str), encoding="utf-8")
            except Exception as exc:
                logger.debug(f"Cache write error: {exc}")

    def invalidate(self, file_path: str) -> None:
        if self._diskcache:
            for k in list(self._diskcache.iterkeys()):
                try:
                    entry = self._diskcache.get(k)
                    if isinstance(entry, dict) and entry.get("_file_path") == file_path:
                        del self._diskcache[k]
                except Exception:
                    pass
        else:
            for f in self.cache_dir.glob("*.json"):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    if data.get("_file_path") == file_path:
                        f.unlink()
                except Exception:
                    pass

    def close(self) -> None:
        if self._diskcache:
            self._diskcache.close()
