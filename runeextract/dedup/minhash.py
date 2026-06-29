"""MinHash LSH and embedding-based document deduplication.

No external dependencies required for MinHash (pure Python implementation).
Embedding-based dedup requires numpy.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Set, Tuple


@dataclass
class DuplicateCluster:
    duplicate_indices: List[int]
    representative_index: int
    similarity_score: float


class MinHashDeduplicator:
    """Near-duplicate detection using MinHash signatures.

    Uses a pure-Python MinHash implementation for computing Jaccard
    similarity between documents based on k-shingles.
    """

    def __init__(self, num_perm: int = 128, shingle_size: int = 5, threshold: float = 0.7):
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self.threshold = threshold
        self._perm_a = self._random_coeffs(num_perm, 2147483647)
        self._perm_b = self._random_coeffs(num_perm, 2147483647)

    @staticmethod
    def _random_coeffs(n: int, mod: int, seed: int = 42) -> List[int]:
        rng = __import__("random").Random(seed)
        return [rng.randint(1, mod - 1) for _ in range(n)]

    def _shingles(self, text: str) -> Set[int]:
        shingles = set()
        for i in range(len(text) - self.shingle_size + 1):
            shingle = text[i:i + self.shingle_size]
            h = int(hashlib.md5(shingle.encode("utf-8")).hexdigest(), 16)
            shingles.add(h)
        return shingles

    def _signature(self, shingles: Set[int]) -> List[int]:
        sig = []
        for a, b in zip(self._perm_a, self._perm_b):
            min_hash = min(((a * s + b) % 2147483647) for s in shingles) if shingles else 0
            sig.append(min_hash)
        return sig

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        sig_a = self._signature(self._shingles(text_a))
        sig_b = self._signature(self._shingles(text_b))
        matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
        return matches / self.num_perm

    def deduplicate(self, texts: List[str]) -> Tuple[List[int], List[DuplicateCluster]]:
        if not texts:
            return [], []

        sigs = [self._signature(self._shingles(t)) for t in texts]
        n = len(texts)
        is_dup = [False] * n
        clusters = []

        for i in range(n):
            if is_dup[i]:
                continue
            dups = []
            for j in range(i + 1, n):
                if is_dup[j]:
                    continue
                matches = sum(1 for a, b in zip(sigs[i], sigs[j]) if a == b)
                sim = matches / self.num_perm
                if sim >= self.threshold:
                    dups.append((j, sim))
            if dups:
                best_dup, best_sim = max(dups, key=lambda x: x[1])
                cluster_indices = [best_dup]
                is_dup[best_dup] = True
                for idx, _ in dups:
                    if idx != best_dup:
                        cluster_indices.append(idx)
                        is_dup[idx] = True
                clusters.append(DuplicateCluster(
                    duplicate_indices=cluster_indices,
                    representative_index=i,
                    similarity_score=best_sim,
                ))

        unique = [i for i in range(n) if not is_dup[i]]
        return unique, clusters


class LSHDeduplicator:
    """Deduplication using Locality-Sensitive Hashing bands.

    More scalable than pairwise MinHash for large collections.
    Uses the same MinHash signatures but buckets documents into
    LSH bands to find candidate pairs efficiently.
    """

    def __init__(self, num_perm: int = 128, num_bands: int = 16, threshold: float = 0.7):
        self.num_perm = num_perm
        self.num_bands = num_bands
        self.rows_per_band = num_perm // num_bands
        self.threshold = threshold
        self._minhash = MinHashDeduplicator(num_perm=num_perm, threshold=threshold)

    def deduplicate(self, texts: List[str]) -> Tuple[List[int], List[DuplicateCluster]]:
        """LSH-based deduplication — faster than :class:`MinHashDeduplicator` when
        the average bucket size is much smaller than the total document count.

        .. note::
           The speed-up over plain MinHash only materializes when documents
           are diverse enough that most LSH buckets contain a small fraction
           of the corpus.  If all documents land in the same few buckets
           (highly similar corpus), this degrades to O(n²) inside each bucket.
        """
        if not texts:
            return [], []

        sigs = [self._minhash._signature(self._minhash._shingles(t)) for t in texts]
        n = len(texts)

        buckets = {}
        for doc_idx, sig in enumerate(sigs):
            for band in range(self.num_bands):
                start = band * self.rows_per_band
                end = start + self.rows_per_band
                band_hash = hash(tuple(sig[start:end]))
                buckets.setdefault((band, band_hash), []).append(doc_idx)

        candidate_pairs = set()
        for (_, _), members in buckets.items():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    candidate_pairs.add((members[i], members[j]) if members[i] < members[j] else (members[j], members[i]))

        is_dup = [False] * n
        clusters = []
        for i in range(n):
            if is_dup[i]:
                continue
            dups = []
            for j in range(i + 1, n):
                if is_dup[j]:
                    continue
                if (i, j) in candidate_pairs:
                    matches = sum(1 for a, b in zip(sigs[i], sigs[j]) if a == b)
                    sim = matches / self.num_perm
                    if sim >= self.threshold:
                        dups.append((j, sim))
            if dups:
                best_dup, best_sim = max(dups, key=lambda x: x[1])
                cluster_indices = [best_dup]
                is_dup[best_dup] = True
                for idx, _ in dups:
                    if idx != best_dup:
                        cluster_indices.append(idx)
                        is_dup[idx] = True
                clusters.append(DuplicateCluster(
                    duplicate_indices=cluster_indices,
                    representative_index=i,
                    similarity_score=best_sim,
                ))

        unique = [i for i in range(n) if not is_dup[i]]
        return unique, clusters


class EmbeddingDeduplicator:
    """Deduplication using embedding vectors and cosine similarity.

    Requires numpy. Use with any embedding provider (AIProcessor, ONNX model, etc.).
    """

    def __init__(self, threshold: float = 0.92):
        self.threshold = threshold

    def deduplicate(self, embeddings: List[List[float]]) -> Tuple[List[int], List[DuplicateCluster]]:
        import numpy as np

        if not embeddings:
            return [], []

        arr = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        arr = arr / norms
        sim_matrix = np.dot(arr, arr.T)

        n = len(embeddings)
        is_dup = [False] * n
        clusters = []

        for i in range(n):
            if is_dup[i]:
                continue
            dups = []
            for j in range(i + 1, n):
                if is_dup[j]:
                    continue
                sim = float(sim_matrix[i][j])
                if sim >= self.threshold:
                    dups.append((j, sim))
            if dups:
                best_dup, best_sim = max(dups, key=lambda x: x[1])
                cluster_indices = [best_dup]
                is_dup[best_dup] = True
                for idx, _ in dups:
                    if idx != best_dup:
                        cluster_indices.append(idx)
                        is_dup[idx] = True
                clusters.append(DuplicateCluster(
                    duplicate_indices=cluster_indices,
                    representative_index=i,
                    similarity_score=best_sim,
                ))

        unique = [i for i in range(n) if not is_dup[i]]
        return unique, clusters


def deduplicate(texts: List[str], strategy: str = "minhash", **kwargs) -> Tuple[List[int], List[DuplicateCluster]]:
    """Deduplicate a list of texts.

    Args:
        texts: List of text strings to deduplicate
        strategy: "minhash" (default), "lsh", or "embedding"
        **kwargs: Passed to the deduplicator constructor

    Returns:
        Tuple of (unique_indices, duplicate_clusters)
    """
    strategies = {
        "minhash": MinHashDeduplicator,
        "lsh": LSHDeduplicator,
        "embedding": EmbeddingDeduplicator,
    }
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy '{strategy}'. Options: {', '.join(strategies)}")

    deduper = strategies[strategy](**kwargs)
    if strategy == "embedding":
        raise ValueError(
            "Use deduplicate_embeddings() for embedding strategy. "
            "deduplicate() expects text input."
        )
    return deduper.deduplicate(texts)


def deduplicate_embeddings(embeddings: List[List[float]], threshold: float = 0.92) -> Tuple[List[int], List[DuplicateCluster]]:
    """Deduplicate using embedding vectors and cosine similarity.

    Args:
        embeddings: List of embedding vectors.
        threshold: Cosine similarity threshold (default 0.92).

    Returns:
        Tuple of (unique_indices, duplicate_clusters)
    """
    return EmbeddingDeduplicator(threshold=threshold).deduplicate(embeddings)


def deduplicate_documents(texts: List[str], strategy: str = "minhash", **kwargs) -> Tuple[List[int], List[DuplicateCluster]]:
    """Alias for deduplicate()."""
    return deduplicate(texts, strategy=strategy, **kwargs)
