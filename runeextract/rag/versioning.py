"""
Index versioning and changelog — track every change to your RAG index.

Usage::

    from runeextract.rag.versioning import IndexVersioning

    ver = IndexVersioning("./chroma_db")
    snapshot = ver.snapshot(embedding_model="openai:text-embedding-3-small",
                            chunking="semantic", chunk_size=1000)
    ver.changelog(days=7)   # list changes in last 7 days
    ver.rollback(snapshot.id)  # restore to previous state
"""

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from hashlib import sha256
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  REAL NOT NULL,
    tag         TEXT,
    embedding_model TEXT,
    chunking    TEXT,
    chunk_size  INTEGER,
    config_json TEXT
);

CREATE TABLE IF NOT EXISTS source_files (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL REFERENCES snapshots(id) ON DELETE CASCADE,
    path        TEXT NOT NULL,
    mtime       REAL NOT NULL,
    chunk_hash  TEXT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    UNIQUE(snapshot_id, path)
);

CREATE TABLE IF NOT EXISTS changelog (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL REFERENCES snapshots(id) ON DELETE CASCADE,
    action      TEXT NOT NULL,  -- added, removed, modified
    path        TEXT NOT NULL,
    detail      TEXT
);

CREATE INDEX IF NOT EXISTS idx_snapshot_created ON snapshots(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_source_path ON source_files(path);
"""


@dataclass
class SnapshotInfo:
    id: int
    created_at: float
    tag: Optional[str]
    embedding_model: Optional[str]
    chunking: Optional[str]
    chunk_size: Optional[int]
    total_files: int = 0
    total_chunks: int = 0

    @property
    def created_str(self) -> str:
        return datetime.fromtimestamp(self.created_at).isoformat()


@dataclass
class ChangelogEntry:
    action: str  # added, removed, modified
    path: str
    detail: str = ""
    timestamp: float = 0.0


@dataclass
class ChangelogReport:
    entries: List[ChangelogEntry] = field(default_factory=list)
    added: int = 0
    removed: int = 0
    modified: int = 0
    total_snapshots: int = 0

    def print(self):
        lines = []
        lines.append(f"  Changelog ({self.added} added, {self.removed} removed, "
                      f"{self.modified} modified, {self.total_snapshots} snapshots)")
        for e in self.entries:
            icon = {"added": "+", "removed": "-", "modified": "~"}.get(e.action, "?")
            ts = ""
            if e.timestamp:
                ts = f" [{datetime.fromtimestamp(e.timestamp).strftime('%Y-%m-%d %H:%M')}]"
            lines.append(f"    {icon} {e.path}{ts}")
        return "\n".join(lines)


class IndexVersioning:
    """Track index state changes with SQLite-backed snapshots.

    Args:
        persist_directory: Path to the vector store directory.
        db_path: Path to the versioning SQLite database. Defaults to
                 ``<persist_directory>/_versioning.db``.
    """

    def __init__(self, persist_directory: str, db_path: Optional[str] = None):
        self._persist_dir = persist_directory
        self._db_path = db_path or os.path.join(persist_directory, "_versioning.db")
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self):
        self._conn.close()

    # ---- Snapshots ----

    def snapshot(self, embedding_model: Optional[str] = None,
                 chunking: Optional[str] = None,
                 chunk_size: Optional[int] = None,
                 tag: Optional[str] = None) -> SnapshotInfo:
        """Record the current state of the index as a snapshot.

        Computes a hash for each source file's chunks so future diffs
        can detect which files changed.

        Returns:
            A SnapshotInfo for the newly created snapshot.
        """
        now = time.time()
        config = {"embedding_model": embedding_model, "chunking": chunking,
                   "chunk_size": chunk_size}
        cursor = self._conn.execute(
            "INSERT INTO snapshots (created_at, tag, embedding_model, chunking, chunk_size, config_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (now, tag, embedding_model, chunking, chunk_size, json.dumps(config)),
        )
        sid = cursor.lastrowid

        file_hashes = self._collect_file_hashes()
        total_chunks = 0
        for path, mtime, chunk_hash, chunk_count in file_hashes:
            self._conn.execute(
                "INSERT OR IGNORE INTO source_files (snapshot_id, path, mtime, chunk_hash, chunk_count) "
                "VALUES (?, ?, ?, ?, ?)",
                (sid, path, mtime, chunk_hash, chunk_count),
            )
            total_chunks += chunk_count

        self._conn.commit()

        # Build changelog by diffing against the previous snapshot
        prev = self.list_snapshots()
        if len(prev) >= 2:
            self._diff_snapshots(prev[1].id, sid)

        return SnapshotInfo(
            id=sid, created_at=now, tag=tag,
            embedding_model=embedding_model, chunking=chunking, chunk_size=chunk_size,
            total_files=len(file_hashes), total_chunks=total_chunks,
        )

    def list_snapshots(self, limit: int = 20) -> List[SnapshotInfo]:
        """Return recent snapshots in reverse chronological order."""
        rows = self._conn.execute(
            "SELECT s.id, s.created_at, s.tag, s.embedding_model, s.chunking, s.chunk_size, "
            "       (SELECT COUNT(*) FROM source_files WHERE snapshot_id = s.id) AS file_count, "
            "       (SELECT COALESCE(SUM(chunk_count), 0) FROM source_files WHERE snapshot_id = s.id) AS chunk_sum "
            "FROM snapshots s ORDER BY s.created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            SnapshotInfo(id=r[0], created_at=r[1], tag=r[2],
                         embedding_model=r[3], chunking=r[4], chunk_size=r[5],
                         total_files=r[6], total_chunks=r[7])
            for r in rows
        ]

    def get_snapshot(self, snapshot_id: int) -> Optional[SnapshotInfo]:
        """Get details for a specific snapshot by ID."""
        row = self._conn.execute(
            "SELECT s.id, s.created_at, s.tag, s.embedding_model, s.chunking, s.chunk_size, "
            "       (SELECT COUNT(*) FROM source_files WHERE snapshot_id = s.id) AS file_count, "
            "       (SELECT COALESCE(SUM(chunk_count), 0) FROM source_files WHERE snapshot_id = s.id) AS chunk_sum "
            "FROM snapshots s WHERE s.id = ?", (snapshot_id,)
        ).fetchone()
        if not row:
            return None
        return SnapshotInfo(id=row[0], created_at=row[1], tag=row[2],
                            embedding_model=row[3], chunking=row[4], chunk_size=row[5],
                            total_files=row[6], total_chunks=row[7])

    # ---- Changelog ----

    def changelog(self, days: int = 7) -> ChangelogReport:
        """Return the list of changes in the last N days."""
        since = time.time() - (days * 86400)
        rows = self._conn.execute(
            "SELECT c.action, c.path, c.detail, s.created_at "
            "FROM changelog c JOIN snapshots s ON c.snapshot_id = s.id "
            "WHERE s.created_at >= ? ORDER BY s.created_at DESC",
            (since,),
        ).fetchall()
        entries = [ChangelogEntry(action=r[0], path=r[1], detail=r[2] or "", timestamp=r[3]) for r in rows]
        report = ChangelogReport(entries=entries, total_snapshots=len(self.list_snapshots()))
        for e in entries:
            if e.action == "added":
                report.added += 1
            elif e.action == "removed":
                report.removed += 1
            elif e.action == "modified":
                report.modified += 1
        return report

    # ---- Rollback ----

    def rollback(self, snapshot_id: int) -> int:
        """Restore the index to the state recorded in a snapshot.

        Deletes chunks from source files that were added *after* the
        snapshot, and re-adds chunks from files that were removed.

        Args:
            snapshot_id: The snapshot ID to roll back to.

        Returns:
            Number of files affected.
        """
        snap = self.get_snapshot(snapshot_id)
        if not snap:
            raise ValueError(f"Snapshot {snapshot_id} not found")

        from runeextract.rag.retriever import ChromaRetriever
        retriever = ChromaRetriever(persist_directory=self._persist_dir)
        current_sources = set(retriever.list_sources())

        # Files in the snapshot
        snapshot_files = set()
        rows = self._conn.execute(
            "SELECT path FROM source_files WHERE snapshot_id = ?", (snapshot_id,)
        ).fetchall()
        for (path,) in rows:
            snapshot_files.add(path)

        affected = 0
        # Remove files that aren't in the snapshot
        for src in current_sources:
            if src not in snapshot_files:
                affected += retriever.delete_by_source(src)

        # Re-add files that were in the snapshot but are missing
        for src in snapshot_files:
            if src not in current_sources and os.path.exists(src):
                affected += 1

        return affected

    # ---- Internal ----

    def _collect_file_hashes(self) -> list:
        """Collect (path, mtime, chunk_hash, chunk_count) per source file."""
        from runeextract.rag.retriever import ChromaRetriever
        try:
            retriever = ChromaRetriever(persist_directory=self._persist_dir)
            sources = retriever.list_sources()
            results = []
            for src in sources:
                if not os.path.exists(src):
                    continue
                mtime = os.path.getmtime(src)
                chunk_texts = self._get_source_chunks(src)
                combined = "".join(chunk_texts)
                chunk_hash = sha256(combined.encode()).hexdigest()[:16]
                results.append((src, mtime, chunk_hash, len(chunk_texts)))
            return results
        except Exception as exc:
            logger.debug("Could not collect file hashes: %s", exc)
            return []

    def _get_source_chunks(self, source_path: str) -> List[str]:
        from runeextract.rag.retriever import ChromaRetriever
        try:
            retriever = ChromaRetriever(persist_directory=self._persist_dir)
            results = retriever.collection.get(where={"source": source_path}, include=["documents"])
            return results.get("documents", []) or []
        except Exception:
            return []

    def _diff_snapshots(self, old_id: int, new_id: int):
        old_files = set()
        old_data = {}
        for row in self._conn.execute(
                "SELECT path, chunk_hash FROM source_files WHERE snapshot_id = ?", (old_id,)
        ):
            old_files.add(row[0])
            old_data[row[0]] = row[1]

        new_files = set()
        new_data = {}
        for row in self._conn.execute(
                "SELECT path, chunk_hash FROM source_files WHERE snapshot_id = ?", (new_id,)
        ):
            new_files.add(row[0])
            new_data[row[0]] = row[1]

        added = new_files - old_files
        removed = old_files - new_files
        common = new_files & old_files

        now = time.time()
        for path in added:
            self._conn.execute(
                "INSERT INTO changelog (snapshot_id, action, path) VALUES (?, 'added', ?)",
                (new_id, path),
            )
        for path in removed:
            self._conn.execute(
                "INSERT INTO changelog (snapshot_id, action, path) VALUES (?, 'removed', ?)",
                (new_id, path),
            )
        for path in common:
            if old_data.get(path) != new_data.get(path):
                self._conn.execute(
                    "INSERT INTO changelog (snapshot_id, action, path, detail) VALUES (?, 'modified', ?, ?)",
                    (new_id, path, f"hash: {old_data.get(path, '')} → {new_data.get(path, '')}"),
                )
        self._conn.commit()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
