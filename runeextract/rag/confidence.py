"""
Multi-factor confidence scoring for RAG answers.

``ConfidenceScorer`` evaluates answer quality from multiple angles:
retrieval score distribution, source diversity, answer faithfulness
(via LLM judge when available), and chunk relevance.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from runeextract.rag.types import ChunkWithScore

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceFactors:
    retrieval_score: float = 0.0
    source_diversity: float = 0.0
    chunk_relevance: float = 0.0
    faithfulness: float = 0.0
    overall: float = 0.0


class ConfidenceScorer:
    """Compute multi-factor confidence for a RAG result.

    Args:
        llm_judge: Optional callable ``fn(answer, context) -> float``
            for LLM-based faithfulness scoring.
    """

    def __init__(self, llm_judge: Optional[Callable[[str, str], float]] = None):
        self._llm_judge = llm_judge

    def score(self, chunks: List[ChunkWithScore],
              answer: str,
              question: str) -> ConfidenceFactors:
        if not chunks:
            return ConfidenceFactors()

        retrieval = self._retrieval_confidence(chunks)
        diversity = self._source_diversity(chunks)
        relevance = self._chunk_relevance(chunks, question)
        faithfulness = self._faithfulness(answer, chunks)

        overall = 0.3 * retrieval + 0.2 * diversity + 0.2 * relevance + 0.3 * faithfulness

        return ConfidenceFactors(
            retrieval_score=retrieval,
            source_diversity=diversity,
            chunk_relevance=relevance,
            faithfulness=faithfulness,
            overall=round(overall, 4),
        )

    @staticmethod
    def _retrieval_confidence(chunks: List[ChunkWithScore]) -> float:
        if not chunks:
            return 0.0
        scores = [c.score for c in chunks]
        avg = sum(scores) / len(scores)
        spread = max(scores) - min(scores) if len(scores) > 1 else 0.0
        penalty = spread * 0.2
        return max(0.0, min(1.0, avg - penalty))

    @staticmethod
    def _source_diversity(chunks: List[ChunkWithScore]) -> float:
        if not chunks:
            return 0.0
        sources = set(c.source for c in chunks if c.source)
        if not sources:
            return 0.3
        ratio = len(sources) / len(chunks)
        return min(1.0, ratio * 2)

    @staticmethod
    def _chunk_relevance(chunks: List[ChunkWithScore], question: str) -> float:
        if not chunks or not question:
            return 0.0
        q_words = set(question.lower().split())
        if not q_words:
            return 0.0
        overlaps = []
        for c in chunks:
            c_words = set(c.text.lower().split())
            overlap = len(q_words & c_words) / len(q_words)
            overlaps.append(overlap)
        return sum(overlaps) / len(overlaps)

    def _faithfulness(self, answer: str, chunks: List[ChunkWithScore]) -> float:
        if not answer:
            return 0.0
        if self._llm_judge:
            try:
                context = "\n".join(c.text[:1000] for c in chunks)
                return self._llm_judge(answer, context)
            except Exception as exc:
                logger.debug("LLM faithfulness judge failed: %s", exc)

        return self._lexical_faithfulness(answer, chunks)

    @staticmethod
    def _lexical_faithfulness(answer: str, chunks: List[ChunkWithScore]) -> float:
        a_words = set(answer.lower().split())
        if not a_words:
            return 0.0
        c_words = set()
        for c in chunks:
            c_words.update(c.text.lower().split())
        if not c_words:
            return 0.0
        supported = sum(1 for w in a_words if w in c_words)
        return supported / len(a_words)
