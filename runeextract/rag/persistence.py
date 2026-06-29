"""AutoRAG state persistence — serialize/restore index state to disk."""

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class IndexState:
    file_hashes: Dict[str, str] = field(default_factory=dict)
    file_chunk_ids: Dict[str, List[str]] = field(default_factory=dict)
    collection_name: str = ""
    persist_directory: str = ""
    embedding_spec: str = ""
    total_documents: int = 0


class IndexPersister:
    def __init__(self, state_dir: Optional[str] = None):
        self.state_dir = state_dir or str(Path.home() / ".runeextract_cache" / "rag_state")
        self._lock = threading.Lock()

    def save(self, state: IndexState, tag: str = "default"):
        os.makedirs(self.state_dir, exist_ok=True)
        path = os.path.join(self.state_dir, f"{tag}.json")
        with self._lock:
            with open(path, "w") as f:
                json.dump({
                    "file_hashes": state.file_hashes,
                    "file_chunk_ids": {k: v for k, v in state.file_chunk_ids.items()},
                    "collection_name": state.collection_name,
                    "persist_directory": state.persist_directory,
                    "embedding_spec": state.embedding_spec,
                    "total_documents": state.total_documents,
                }, f, indent=2)

    def load(self, tag: str = "default") -> Optional[IndexState]:
        path = os.path.join(self.state_dir, f"{tag}.json")
        if not os.path.exists(path):
            return None
        try:
            with self._lock:
                with open(path) as f:
                    data = json.load(f)
            return IndexState(
                file_hashes=data.get("file_hashes", {}),
                file_chunk_ids=data.get("file_chunk_ids", {}),
                collection_name=data.get("collection_name", ""),
                persist_directory=data.get("persist_directory", ""),
                embedding_spec=data.get("embedding_spec", ""),
                total_documents=data.get("total_documents", 0),
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load index state: %s", e)
            return None

    def list_snapshots(self) -> List[str]:
        os.makedirs(self.state_dir, exist_ok=True)
        return sorted(
            f.replace(".json", "")
            for f in os.listdir(self.state_dir)
            if f.endswith(".json")
        )

    def delete(self, tag: str = "default"):
        path = os.path.join(self.state_dir, f"{tag}.json")
        if os.path.exists(path):
            os.remove(path)

    def save_to_rag(self, rag, tag: str = "default"):
        state = IndexState(
            file_hashes=getattr(rag, "_file_hashes", {}),
            file_chunk_ids=getattr(rag, "_file_chunk_ids", {}),
            collection_name=getattr(rag, "collection_name", ""),
            persist_directory=getattr(rag, "persist_directory", ""),
            embedding_spec=str(getattr(rag, "embedding_spec", "")),
            total_documents=len(getattr(rag, "_documents", [])),
        )
        self.save(state, tag=tag)

    def restore_to_rag(self, rag, tag: str = "default"):
        state = self.load(tag)
        if state is None:
            logger.info("No saved state found for tag '%s'", tag)
            return False
        rag._file_hashes = state.file_hashes
        rag._file_chunk_ids = state.file_chunk_ids
        logger.info("Restored state for %d files (tag=%s)", len(state.file_hashes), tag)
        return True
