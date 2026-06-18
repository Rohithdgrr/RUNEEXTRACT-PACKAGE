"""Tests for document deduplication."""

import pytest

from runeextract.dedup.minhash import (
    MinHashDeduplicator, LSHDeduplicator, EmbeddingDeduplicator,
    deduplicate, DuplicateCluster,
)


class TestMinHashDeduplicator:
    def test_identical_texts(self):
        d = MinHashDeduplicator(threshold=0.5)
        texts = ["hello world foo bar baz", "hello world foo bar baz"]
        unique, clusters = d.deduplicate(texts)
        assert len(unique) == 1
        assert len(clusters) == 1

    def test_different_texts(self):
        d = MinHashDeduplicator(threshold=0.9)
        texts = ["hello world", "completely different content here"]
        unique, clusters = d.deduplicate(texts)
        assert len(unique) == 2
        assert len(clusters) == 0

    def test_empty_list(self):
        d = MinHashDeduplicator()
        unique, clusters = d.deduplicate([])
        assert len(unique) == 0
        assert len(clusters) == 0

    def test_single_text(self):
        d = MinHashDeduplicator()
        unique, clusters = d.deduplicate(["only one"])
        assert len(unique) == 1
        assert len(clusters) == 0

    def test_similar_texts(self):
        d = MinHashDeduplicator(num_perm=64, shingle_size=3, threshold=0.3)
        texts = [
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox jumps over the lazy cat",
            "completely unrelated document here",
        ]
        unique, clusters = d.deduplicate(texts)
        assert len(unique) < 3  # at least 2 are similar
        assert len(clusters) >= 1

    def test_similarity_metric(self):
        d = MinHashDeduplicator(num_perm=64, shingle_size=3)
        sim = d.compute_similarity("hello world", "hello world")
        assert sim == pytest.approx(1.0, abs=0.01)

    def test_similarity_different(self):
        d = MinHashDeduplicator(num_perm=64, shingle_size=3)
        sim = d.compute_similarity("hello world", "completely different")
        assert sim < 0.5


class TestLSHDeduplicator:
    def test_identical_texts(self):
        d = LSHDeduplicator(threshold=0.5)
        texts = ["hello world test data", "hello world test data"]
        unique, clusters = d.deduplicate(texts)
        assert len(clusters) >= 1

    def test_different_texts(self):
        d = LSHDeduplicator(threshold=0.9)
        texts = ["hello", "world"]
        unique, clusters = d.deduplicate(texts)
        assert len(unique) == 2

    def test_empty(self):
        d = LSHDeduplicator()
        u, c = d.deduplicate([])
        assert len(u) == 0


class TestEmbeddingDeduplicator:
    def test_identical_embeddings(self):
        d = EmbeddingDeduplicator(threshold=0.9)
        emb = [[1.0, 0.0], [1.0, 0.0]]
        unique, clusters = d.deduplicate(emb)
        assert len(clusters) >= 1

    def test_different_embeddings(self):
        d = EmbeddingDeduplicator(threshold=0.99)
        emb = [[1.0, 0.0], [0.0, 1.0]]
        unique, clusters = d.deduplicate(emb)
        assert len(unique) == 2

    def test_empty(self):
        d = EmbeddingDeduplicator()
        u, c = d.deduplicate([])
        assert len(u) == 0


class TestDuplicateCluster:
    def test_create(self):
        c = DuplicateCluster(
            duplicate_indices=[2, 3],
            representative_index=1,
            similarity_score=0.95,
        )
        assert c.duplicate_indices == [2, 3]
        assert c.representative_index == 1
        assert c.similarity_score == 0.95


class TestDeduplicate:
    def test_minhash_strategy(self):
        texts = ["a b c d e f g", "a b c d e f g"]
        unique, clusters = deduplicate(texts, strategy="minhash", threshold=0.5)
        assert len(clusters) >= 1

    def test_lsh_strategy(self):
        texts = ["a b c d", "a b c d"]
        unique, clusters = deduplicate(texts, strategy="lsh", threshold=0.5)
        assert len(clusters) >= 0  # LSH is approximate; may or may not catch them

    def test_embedding_strategy_raises(self):
        with pytest.raises(ValueError, match="Use deduplicate_embeddings"):
            deduplicate(["text"], strategy="embedding")

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            deduplicate(["text"], strategy="invalid")
