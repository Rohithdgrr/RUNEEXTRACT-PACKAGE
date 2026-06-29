"""Persistent embedding cache — disk-backed, survives restarts.

Maps (text_hash, model_name) -> embedding vector via SQLite.
"""

import hashlib
import json
import logging
import os
import sqlite3
import struct
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class PersistentEmbeddingCache:
    def __init__(self, db_path: Optional[str] = None, max_entries: int = 100_000):
        self.db_path = db_path or str(Path.home() / ".runeextract_cache" / "embeddings.db")
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "  text_hash TEXT NOT NULL,"
            "  model TEXT NOT NULL,"
            "  dim INTEGER NOT NULL,"
            "  vector BLOB NOT NULL,"
            "  created_at REAL NOT NULL,"
            "  access_count INTEGER DEFAULT 0,"
            "  PRIMARY KEY (text_hash, model)"
            ")"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_access "
            "ON embeddings(access_count)"
        )
        self._conn.commit()

    def _make_key(self, text: str, model: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _pack_vector(self, vec: List[float]) -> bytes:
        return struct.pack(f"{len(vec)}f", *vec)

    def _unpack_vector(self, data: bytes, dim: int) -> List[float]:
        return list(struct.unpack(f"{dim}f", data))

    def get(self, text: str, model: str) -> Optional[List[float]]:
        key = self._make_key(text, model)
        with self._lock:
            cur = self._conn.execute(
                "SELECT vector, dim FROM embeddings WHERE text_hash=? AND model=?",
                (key, model),
            )
            row = cur.fetchone()
            if row:
                self._conn.execute(
                    "UPDATE embeddings SET access_count = access_count + 1 WHERE text_hash=? AND model=?",
                    (key, model),
                )
                self._conn.commit()
                return self._unpack_vector(row[0], row[1])
        return None

    def put(self, text: str, model: str, vector: List[float]):
        key = self._make_key(text, model)
        dim = len(vector)
        blob = self._pack_vector(vector)
        with self._lock:
            cur = self._conn.execute(
                "SELECT 1 FROM embeddings WHERE text_hash=? AND model=?",
                (key, model),
            )
            if cur.fetchone() is None:
                self._conn.execute(
                    "SELECT COUNT(*) FROM embeddings"
                )
                count = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
                if count >= self.max_entries:
                    self._evict_lru()
                self._conn.execute(
                    "INSERT OR REPLACE INTO embeddings (text_hash, model, dim, vector, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (key, model, dim, blob, __import__("time").time()),
                )
                self._conn.commit()

    def get_batch(self, texts: List[str], model: str) -> Dict[str, Optional[List[float]]]:
        result = {}
        for text in texts:
            result[text] = self.get(text, model)
        return result

    def put_batch(self, texts: List[str], model: str, vectors: List[List[float]]):
        for text, vec in zip(texts, vectors):
            self.put(text, model, vec)

    def _evict_lru(self):
        self._conn.execute(
            "DELETE FROM embeddings WHERE rowid IN ("
            "  SELECT rowid FROM embeddings ORDER BY access_count ASC, created_at ASC LIMIT 1000"
            ")"
        )
        self._conn.commit()

    def clear(self):
        with self._lock:
            self._conn.execute("DELETE FROM embeddings")
            self._conn.commit()

    @property
    def size(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
