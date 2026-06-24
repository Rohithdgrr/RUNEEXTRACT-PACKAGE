"""
Contextual compression for fitting retrieved chunks into a token budget.
"""

import logging
from typing import List, Optional

from runeextract.rag.types import ChunkWithScore
from runeextract.utils.maturity import beta

logger = logging.getLogger(__name__)


@beta(name="rag.compressor")
class ContextualCompressor:
    """Compress retrieved chunks to fit within a token budget.

    Uses iterative strategies: per-chunk summarization → adjacent merge
    → query-aware sentence selection.
    """

    def __init__(self, llm_complete: Optional[callable] = None,
                 max_tokens: int = 4000):
        self.llm_complete = llm_complete
        self.max_tokens = max_tokens

    def compress(self, chunks: List[ChunkWithScore], query: str,
                 max_tokens: Optional[int] = None) -> List[ChunkWithScore]:
        """Reduce chunks to fit within the token budget.

        Strategies applied in order:
          1. Summarize long per-chunk text
          2. Merge adjacent chunks from the same source
          3. Query-aware truncation of low-relevance sentences
        """
        budget = max_tokens or self.max_tokens
        total = sum(self._estimate_tokens(c.text) for c in chunks)

        if total <= budget:
            return chunks

        chunks = self._summarize_long_chunks(chunks, query)
        total = sum(self._estimate_tokens(c.text) for c in chunks)
        if total <= budget:
            return chunks

        merged = self._merge_adjacent(chunks)
        total = sum(self._estimate_tokens(c.text) for c in merged)
        if total <= budget:
            return merged

        return self._truncate_by_relevance(merged, query, budget)

    def _summarize_long_chunks(self, chunks: List[ChunkWithScore],
                               query: str) -> List[ChunkWithScore]:
        """Summarize chunks exceeding a threshold."""
        if not self.llm_complete:
            return chunks
        result = []
        for c in chunks:
            if self._estimate_tokens(c.text) > 600:
                prompt = (
                    f"Summarize the following text keeping facts relevant to: {query}\n\n"
                    f"{c.text[:2000]}"
                )
                try:
                    summary = self.llm_complete(prompt, max_tokens=200)
                    c.text = summary.strip()
                except Exception as exc:
                    logger.warning("Chunk summarization failed: %s", exc)
            result.append(c)
        return result

    def _merge_adjacent(self, chunks: List[ChunkWithScore]) -> List[ChunkWithScore]:
        """Merge consecutive chunks from the same document source."""
        if not chunks:
            return chunks
        merged = []
        current = chunks[0]
        for nxt in chunks[1:]:
            if (current.source == nxt.source and
                    self._estimate_tokens(current.text) + self._estimate_tokens(nxt.text) < 800):
                current.text += "\n" + nxt.text
            else:
                merged.append(current)
                current = nxt
        merged.append(current)
        return merged

    def _truncate_by_relevance(self, chunks: List[ChunkWithScore],
                               query: str, budget: int) -> List[ChunkWithScore]:
        """Sort chunks by score and keep as many as fit in budget."""
        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
        result = []
        used = 0
        for c in sorted_chunks:
            tokens = self._estimate_tokens(c.text)
            if used + tokens > budget:
                continue
            result.append(c)
            used += tokens
        return result

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return int(len(text.split()) * 1.33)
