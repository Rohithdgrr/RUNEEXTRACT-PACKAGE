"""RuneExtract Dedup — document deduplication using MinHash LSH and embeddings."""

from runeextract.dedup.minhash import (
    MinHashDeduplicator, LSHDeduplicator, EmbeddingDeduplicator,
    deduplicate, deduplicate_documents, deduplicate_embeddings,
)

__all__ = [
    "MinHashDeduplicator", "LSHDeduplicator", "EmbeddingDeduplicator",
    "deduplicate", "deduplicate_documents", "deduplicate_embeddings",
]
