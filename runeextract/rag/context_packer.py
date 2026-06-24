"""
Smart context packing for RAG answer generation.

``ContextPacker`` fits retrieved chunks into an LLM's context window
while maximising relevant information through configurable strategies:
sorted by relevance, compressed via summarisation, or source-grouped.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from runeextract.rag.types import ChunkWithScore

logger = logging.getLogger(__name__)


@dataclass
class PackedContext:
    text: str
    chunks_used: int
    total_tokens: int
    strategy: str
    chunk_map: Dict[int, int] = field(default_factory=dict)


class ContextPacker:
    """Pack retrieved chunks into a token budget.

    Args:
        max_tokens: Maximum tokens for the packed context.
        token_estimator: Optional callable ``fn(text) -> int``. Defaults to
            ``len(text.split()) * 1.33``.
    """

    def __init__(self, max_tokens: int = 4000,
                 token_estimator: Optional[Callable[[str], int]] = None):
        self.max_tokens = max_tokens
        self._token_estimate = token_estimator or self._default_estimate

    @staticmethod
    def _default_estimate(text: str) -> int:
        return int(len(text.split()) * 1.33)

    def pack(self, chunks: List[ChunkWithScore], query: str,
             strategy: str = "sorted",
             max_tokens: Optional[int] = None) -> PackedContext:
        """Pack chunks into context within the token budget.

        Strategies:
            * ``"sorted"`` — sort by relevance score descending, take until budget full.
            * ``"compressed"`` — summarise low-score chunks when budget is exceeded.
            * ``"structured"`` — group by source, interleave high-score chunks per source.
        """
        budget = max_tokens or self.max_tokens
        strategy_map = {
            "sorted": self._pack_sorted,
            "compressed": self._pack_compressed,
            "structured": self._pack_structured,
        }
        pack_fn = strategy_map.get(strategy, self._pack_sorted)
        return pack_fn(chunks, query, budget)

    def _pack_sorted(self, chunks: List[ChunkWithScore], query: str,
                     budget: int) -> PackedContext:
        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
        return self._take_until_budget(sorted_chunks, budget, "sorted")

    def _pack_compressed(self, chunks: List[ChunkWithScore], query: str,
                         budget: int) -> PackedContext:
        chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
        total = sum(self._token_estimate(c.text) for c in chunks)
        if total <= budget:
            return self._take_until_budget(chunks, budget, "compressed")

        keep = []
        summary_candidates = []
        for c in chunks:
            if c.score >= 0.5:
                keep.append(c)
            else:
                summary_candidates.append(c)

        if summary_candidates:
            combined = "\n".join(c.text[:500] for c in summary_candidates)
            summary_text = combined[:1000]
            keep.append(ChunkWithScore(
                text=f"[Summarised {len(summary_candidates)} lower-relevance chunks]: {summary_text}",
                score=0.3,
                source="",
            ))

        return self._take_until_budget(keep, budget, "compressed")

    def _pack_structured(self, chunks: List[ChunkWithScore], query: str,
                         budget: int) -> PackedContext:
        groups: Dict[str, List[ChunkWithScore]] = {}
        for c in chunks:
            src = c.source or "unknown"
            groups.setdefault(src, []).append(c)

        interleaved = []
        while any(groups.values()):
            for src in list(groups.keys()):
                if groups[src]:
                    interleaved.append(groups[src].pop(0))

        return self._take_until_budget(interleaved, budget, "structured")

    def _take_until_budget(self, chunks: List[ChunkWithScore], budget: int,
                           strategy: str) -> PackedContext:
        parts: List[str] = []
        used = 0
        chunk_map: Dict[int, int] = {}
        for i, c in enumerate(chunks):
            tokens = self._token_estimate(c.text)
            if used + tokens > budget:
                continue
            parts.append(c.text)
            chunk_map[len(parts) - 1] = i
            used += tokens

        return PackedContext(
            text="\n\n".join(parts),
            chunks_used=len(parts),
            total_tokens=used,
            strategy=strategy,
            chunk_map=chunk_map,
        )
