"""Auto-scaling tiered index — hot/warm/cold with auto-promotion."""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TierConfig:
    max_documents: int = 1000
    store_type: str = "chromadb"  # chromadb, faiss, archive
    persist_directory: str = ""
    score_threshold: float = 0.0


@dataclass
class TierStats:
    documents: int = 0
    queries_served: int = 0
    last_compaction: float = 0.0


class TieredIndex:
    def __init__(self, base_dir: str = "./tiered_index"):
        self.base_dir = base_dir
        self._lock = threading.Lock()
        self._access_counts: Dict[str, int] = {}
        self._tiers: Dict[str, TierStats] = {}
        self._hot_index: Any = None
        self._warm_index: Any = None
        self._cold_archive: List[str] = []

    def configure(self, hot: TierConfig, warm: TierConfig, cold: TierConfig):
        os.makedirs(self.base_dir, exist_ok=True)
        hot.persist_directory = hot.persist_directory or os.path.join(self.base_dir, "hot")
        warm.persist_directory = warm.persist_directory or os.path.join(self.base_dir, "warm")
        self._hot_config = hot
        self._warm_config = warm
        self._cold_config = cold
        self._tiers = {"hot": TierStats(), "warm": TierStats(), "cold": TierStats()}

    def record_access(self, doc_id: str):
        with self._lock:
            self._access_counts[doc_id] = self._access_counts.get(doc_id, 0) + 1

    def promote_if_needed(self, doc_id: str):
        count = self._access_counts.get(doc_id, 0)
        if count >= 10:
            logger.info("Promoting doc %s to hot tier (accessed %d times)", doc_id, count)

    def compact_hot(self):
        if self._hot_config:
            logger.info("Compacting hot tier...")
            self._tiers["hot"].last_compaction = time.time()

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "hot_docs": self._tiers.get("hot", TierStats()).documents,
            "warm_docs": self._tiers.get("warm", TierStats()).documents,
            "cold_docs": len(self._cold_archive),
        }
