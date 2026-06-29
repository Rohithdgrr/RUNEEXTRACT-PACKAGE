"""
Knowledge Deduplication Engine — detect and merge duplicate chunks across a corpus.

Usage::

    from runeextract.rag.dedup_engine import DedupEngine

    engine = DedupEngine(strategy="minhash", threshold=0.85)
    duplicates = engine.find_duplicates(["doc1.pdf", "doc2.pdf"])
    engine.report(duplicates)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from runeextract.dedup.minhash import MinHashDeduplicator, LSHDeduplicator, EmbeddingDeduplicator


logger = logging.getLogger(__name__)


@dataclass
class DuplicateGroup:
    """A group of near-duplicate chunks across documents."""
    chunks: List[Dict] = field(default_factory=list)
    source_files: List[str] = field(default_factory=list)
    similarity: float = 0.0
    representative: Optional[str] = None


@dataclass
class DedupReport:
    groups: List[DuplicateGroup] = field(default_factory=list)
    total_duplicates: int = 0
    total_chunks_scanned: int = 0
    savings_estimate: int = 0  # estimated chunks that could be removed

    def print(self) -> str:
        lines = [
            f"  Dedup Report: {self.total_duplicates} duplicate group(s) found",
            f"  Scanned: {self.total_chunks_scanned} chunks",
            f"  Estimated savings: {self.savings_estimate} redundant chunk(s)",
        ]
        for g in self.groups[:10]:
            lines.append(f"    • {g.similarity:.2%} similar — {', '.join(g.source_files[:3])}")
            if len(g.source_files) > 3:
                lines.append(f"      ... and {len(g.source_files) - 3} more files")
        return "\n".join(lines)


class DedupEngine:
    """Corpus-level duplicate detection across indexed documents.

    Builds on the existing dedup/minhash.py but works at the
    corpus level — loading chunks from the vector store and
    detecting cross-document duplicates.

    Args:
        strategy: ``"minhash"``, ``"lsh"``, or ``"embedding"``.
        threshold: Similarity threshold (default 0.85).
        **kwargs: Passed to the deduplicator constructor.
    """

    def __init__(self, strategy: str = "minhash", threshold: float = 0.85, **kwargs):
        self._strategy = strategy
        self._threshold = threshold
        self._kwargs = kwargs

        if strategy == "minhash":
            self._deduper = MinHashDeduplicator(threshold=threshold, **kwargs)
        elif strategy == "lsh":
            self._deduper = LSHDeduplicator(threshold=threshold, **kwargs)
        elif strategy == "embedding":
            self._deduper = EmbeddingDeduplicator(threshold=threshold)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def find_duplicates(self, texts: List[str],
                        source_map: Optional[Dict[int, str]] = None) -> DedupReport:
        """Find near-duplicate chunks in a list of texts.

        Args:
            texts: List of chunk text strings.
            source_map: Optional mapping from index to source file path.

        Returns:
            A DedupReport with duplicate groups.
        """
        if not texts:
            return DedupReport()

        logger.info("Scanning %d chunks for duplicates (strategy=%s, threshold=%.2f)",
                     len(texts), self._strategy, self._threshold)

        if self._strategy == "embedding":
            unique, clusters = self._deduper.deduplicate(texts)
        else:
            unique, clusters = self._deduper.deduplicate(texts)

        groups = []
        total_duplicates = 0
        for cluster in clusters:
            chunk_texts = []
            source_files = set()
            for idx in cluster.duplicate_indices:
                if idx < len(texts):
                    chunk_texts.append(texts[idx])
                    if source_map and idx in source_map:
                        source_files.add(source_map[idx])
            # Add the representative
            rep_idx = cluster.representative_index
            if rep_idx < len(texts):
                rep_text = texts[rep_idx]
            else:
                rep_text = ""

            groups.append(DuplicateGroup(
                chunks=[{"index": rep_idx, "text": rep_text}] +
                        [{"index": i, "text": texts[i]} for i in cluster.duplicate_indices if i < len(texts)],
                source_files=sorted(source_files),
                similarity=cluster.similarity_score,
                representative=rep_text[:100] if rep_text else None,
            ))
            total_duplicates += len(cluster.duplicate_indices)

        return DedupReport(
            groups=groups,
            total_duplicates=total_duplicates,
            total_chunks_scanned=len(texts),
            savings_estimate=total_duplicates,
        )

    def find_in_index(self, persist_directory: str,
                      collection_name: str = "documents",
                      max_chunks: int = 5000) -> DedupReport:
        """Find duplicates directly from a vector store index.

        Args:
            persist_directory: Path to the ChromaDB persist directory.
            collection_name: Collection name.
            max_chunks: Maximum chunks to scan (default 5000).

        Returns:
            A DedupReport with duplicate groups.
        """
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            logger.error("chromadb required. Install: pip install chromadb")
            return DedupReport()

        client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_or_create_collection(name=collection_name)
        all_data = collection.get(include=["documents", "metadatas"])
        docs = (all_data.get("documents", []) or [])[:max_chunks]
        metas = (all_data.get("metadatas", []) or [])[:max_chunks]

        source_map = {}
        for i, m in enumerate(metas):
            src = (m or {}).get("source", "")
            if src:
                source_map[i] = src

        return self.find_duplicates(docs, source_map=source_map)
