"""
Adaptive hybrid search combining dense (embedding) and sparse (BM25) retrieval.

``HybridSearch`` automatically weights dense vs. sparse contributions
based on the query's lexical density and document characteristics.
"""

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from runeextract.rag.types import ChunkWithScore

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    chunks: List[ChunkWithScore] = field(default_factory=list)
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    query_analysis: Dict[str, float] = field(default_factory=dict)


class BM25Sparse:
    """Lightweight in-memory BM25 implementation — no external dependency."""

    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self._k1 = k1
        self._b = b
        self._corpus_size = len(corpus)
        self._avg_doc_len = 0.0
        self._doc_freq: Dict[str, int] = {}
        self._doc_lens: List[int] = []
        self._build(corpus)

    def _build(self, corpus: List[str]) -> None:
        total_len = 0
        for doc in corpus:
            tokens = self._tokenize(doc)
            self._doc_lens.append(len(tokens))
            total_len += len(tokens)
            seen = set()
            for t in tokens:
                if t not in seen:
                    self._doc_freq[t] = self._doc_freq.get(t, 0) + 1
                    seen.add(t)
        self._avg_doc_len = total_len / self._corpus_size if self._corpus_size else 1.0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def score(self, query: str, doc_idx: int) -> float:
        query_tokens = self._tokenize(query)
        doc_text = ""  # not needed; we use precomputed stats
        doc_len = self._doc_lens[doc_idx] if doc_idx < len(self._doc_lens) else 0
        score = 0.0
        for qt in query_tokens:
            df = self._doc_freq.get(qt, 0)
            if df == 0:
                continue
            idf = math.log((self._corpus_size - df + 0.5) / (df + 0.5) + 1.0)
            score += idf * (self._k1 + 1) / (1 + self._k1 * (1 - self._b + self._b * doc_len / self._avg_doc_len))
        return score


class HybridSearch:
    """Combine dense (embedding) and sparse (BM25) retrieval with adaptive weights.

    Args:
        dense_fn: Callable that takes a query string and returns an embedding list.
        chunks: Reference list of ChunkWithScore for BM25 corpus.
    """

    def __init__(self, dense_fn: Callable[[str], List[float]],
                 chunks: Optional[List[ChunkWithScore]] = None,
                 embeddings: Optional[List[List[float]]] = None):
        self._dense_fn = dense_fn
        self._corpus: List[str] = [c.text for c in chunks] if chunks else []
        self._embeddings: List[List[float]] = embeddings if embeddings else []
        self._bm25: Optional[BM25Sparse] = None
        if self._corpus:
            self._bm25 = BM25Sparse(self._corpus)

    def update_corpus(self, chunks: List[ChunkWithScore],
                      embeddings: Optional[List[List[float]]] = None) -> None:
        self._corpus = [c.text for c in chunks]
        self._embeddings = embeddings if embeddings else []
        self._bm25 = BM25Sparse(self._corpus) if self._corpus else None

    def analyze_query(self, query: str) -> Dict[str, float]:
        """Analyze query characteristics for weight tuning.

        Returns:
            Dict with keys: ``lexical_density``, ``term_count``, ``avg_term_len``.
        """
        tokens = re.findall(r"\w+", query.lower())
        if not tokens:
            return {"lexical_density": 0.5, "term_count": 0, "avg_term_len": 0}
        unique = len(set(tokens))
        avg_len = sum(len(t) for t in tokens) / len(tokens)
        return {
            "lexical_density": unique / len(tokens) if tokens else 0.5,
            "term_count": len(tokens),
            "avg_term_len": avg_len,
        }

    def compute_weights(self, query: str) -> Tuple[float, float]:
        """Adaptively compute dense and sparse weights based on query.

        More sparse weight when query has high lexical density (many
        distinct keywords); more dense weight for short or abstract queries.
        """
        analysis = self.analyze_query(query)
        ld = analysis["lexical_density"]
        tc = analysis["term_count"]
        sparse = 0.3 + 0.5 * ld
        if tc <= 2:
            sparse = 0.2
        dense = 1.0 - sparse
        return dense, sparse

    def search(self, query: str, top_k: int = 5,
               dense_weight: Optional[float] = None,
               sparse_weight: Optional[float] = None) -> HybridResult:
        """Run hybrid search with optional weight override."""
        if dense_weight is None or sparse_weight is None:
            dw, sw = self.compute_weights(query)
            dense_weight = dw if dense_weight is None else dense_weight
            sparse_weight = sw if sparse_weight is None else sparse_weight

        analysis = self.analyze_query(query)
        dense_scores: Dict[int, float] = {}
        sparse_scores: Dict[int, float] = {}

        if self._dense_fn and self._embeddings and len(self._embeddings) == len(self._corpus):
            try:
                query_emb = self._dense_fn(query)
                query_vec = np.array(query_emb, dtype=np.float32)
                chunk_vecs = np.array(self._embeddings, dtype=np.float32)
                norms = np.linalg.norm(chunk_vecs, axis=1) * np.linalg.norm(query_vec)
                sims = np.dot(chunk_vecs, query_vec) / np.maximum(norms, 1e-10)
                dense_scores = {i: float(sim) for i, sim in enumerate(sims)}
            except Exception as exc:
                logger.debug("Dense search failed: %s", exc)

        if self._bm25 and self._corpus:
            for i in range(len(self._corpus)):
                sparse_scores[i] = self._bm25.score(query, i)

        combined: Dict[int, float] = {}
        all_indices = set(dense_scores) | set(sparse_scores)
        for i in all_indices:
            d = dense_scores.get(i, 0.0)
            s = sparse_scores.get(i, 0.0)
            combined[i] = dense_weight * d + sparse_weight * s

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]

        result_chunks = []
        for idx, score in ranked:
            if idx < len(self._corpus):
                result_chunks.append(ChunkWithScore(
                    text=self._corpus[idx],
                    score=float(score),
                    source="",
                ))

        return HybridResult(
            chunks=result_chunks,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            query_analysis=analysis,
        )

    @staticmethod
    def reciprocal_rank_fusion(
        dense_chunks: List[ChunkWithScore],
        sparse_chunks: List[ChunkWithScore],
        k: int = 60,
        top_k: int = 5,
    ) -> List[ChunkWithScore]:
        """Fuse dense and sparse results using Reciprocal Rank Fusion."""
        scores: Dict[str, float] = {}
        text_map: Dict[str, ChunkWithScore] = {}

        for rank, c in enumerate(dense_chunks):
            scores[c.text] = scores.get(c.text, 0.0) + 1.0 / (k + rank)
            text_map[c.text] = c

        for rank, c in enumerate(sparse_chunks):
            scores[c.text] = scores.get(c.text, 0.0) + 1.0 / (k + rank)
            if c.text not in text_map:
                text_map[c.text] = c

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        result = []
        for text, score in ranked:
            c = text_map[text]
            c.score = float(score)
            result.append(c)
        return result
