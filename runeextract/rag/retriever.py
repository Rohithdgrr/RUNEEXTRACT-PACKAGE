"""
Vector store retriever abstraction.

Wraps ChromaDB and FAISS persistent stores for query-time retrieval.
"""

import logging
from typing import List, Optional, Dict, Any, Callable

from runeextract.rag.types import ChunkWithScore
from runeextract.utils.maturity import experimental, beta

logger = logging.getLogger(__name__)


class ChromaRetriever:
    """Query a persisted ChromaDB collection."""

    def __init__(self, persist_directory: str = "./chroma_db",
                 collection_name: str = "documents",
                 embedding_function: Optional[Callable] = None):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self._collection = None

    @property
    def collection(self):
        if self._collection is not None:
            return self._collection
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("chromadb is required. Install: pip install chromadb")
        client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
        )
        return self._collection

    def query(self, query_embedding: List[float], top_k: int = 5,
              metadata_filter: Optional[Dict[str, Any]] = None) -> List[ChunkWithScore]:
        """Search by embedding vector."""
        where = metadata_filter or None
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )
        return self._results_to_chunks(results)

    def query_text(self, query: str, top_k: int = 5,
                   metadata_filter: Optional[Dict[str, Any]] = None) -> List[ChunkWithScore]:
        """Search by text string (ChromaDB handles embedding internally)."""
        where = metadata_filter or None
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where,
        )
        return self._results_to_chunks(results)

    def _results_to_chunks(self, results) -> List[ChunkWithScore]:
        chunks = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return chunks
        for i, doc_id in enumerate(results["ids"][0]):
            meta = (results["metadatas"][0][i] or {}) if results.get("metadatas") else {}
            score = float(results["distances"][0][i]) if results.get("distances") else 0.0
            chunks.append(ChunkWithScore(
                text=results["documents"][0][i] if results.get("documents") else "",
                score=1.0 - score,
                source=meta.get("source", ""),
                source_type=meta.get("source_type", ""),
                document_id=meta.get("document_id", ""),
                chunk_id=meta.get("chunk_id", doc_id),
                page=meta.get("page"),
                metadata=meta,
                char_start=meta.get("start_index"),
                char_end=meta.get("end_index"),
            ))
        return chunks

    def delete_by_source(self, source_path: str) -> int:
        """Delete all entries with matching source path.

        Args:
            source_path: The file path to match against the "source" metadata field.

        Returns:
            Number of entries deleted.
        """
        try:
            results = self.collection.get(where={"source": source_path})
            ids = results.get("ids", [])
            if ids:
                self.collection.delete(ids=ids)
                logger.info("Deleted %d chunks for source: %s", len(ids), source_path)
            return len(ids)
        except Exception as exc:
            logger.warning("delete_by_source failed for %s: %s", source_path, exc)
            return 0

    def list_sources(self) -> List[str]:
        """Return all unique source paths in the collection."""
        try:
            results = self.collection.get(include=["metadatas"])
            sources = set()
            for meta in results.get("metadatas", []):
                src = meta.get("source", "") if meta else ""
                if src:
                    sources.add(src)
            return list(sources)
        except Exception as exc:
            logger.warning("list_sources failed: %s", exc)
            return []


@experimental(name="rag.faiss")
class FAISSRetriever:
    """Query a persisted FAISS index with real embeddings."""

    def __init__(self, index_path: str = "./faiss_index",
                 embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None):
        self.index_path = index_path
        self.embedding_fn = embedding_fn
        self._index = None
        self._metadata = None

    def _load(self):
        if self._index is not None:
            return
        try:
            import faiss
            import json
            import os
        except ImportError:
            raise ImportError("faiss and numpy are required. Install: pip install faiss-cpu numpy")

        index_file = self.index_path + ".index"
        meta_file = self.index_path + ".meta.json"
        old_meta = self.index_path + ".meta.pkl"
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"FAISS index not found at {self.index_path}")
        if not os.path.exists(meta_file) and os.path.exists(old_meta):
            meta_file = old_meta
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"FAISS metadata not found at {self.index_path}")
        self._index = faiss.read_index(index_file)
        if meta_file.endswith(".pkl"):
            import pickle
            with open(meta_file, "rb") as f:
                self._metadata = pickle.load(f)
        else:
            with open(meta_file, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)

    @property
    def index(self):
        self._load()
        return self._index

    @property
    def metadata(self):
        self._load()
        return self._metadata

    def query(self, query_embedding: List[float], top_k: int = 5) -> List[ChunkWithScore]:
        """Search by embedding vector."""
        import numpy as np
        query_vec = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_vec, top_k)
        chunks = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            chunks.append(ChunkWithScore(
                text=meta.get("text", ""),
                score=float(1.0 - distances[0][i] / max(distances[0]) if distances[0].max() > 0 else 0.0),
                source=meta.get("source", ""),
                source_type=meta.get("source_type", ""),
                document_id=meta.get("document_id", ""),
                chunk_id=meta.get("chunk_id", ""),
                page=meta.get("page"),
                metadata=meta,
                char_start=meta.get("start_index"),
                char_end=meta.get("end_index"),
            ))
        return chunks
