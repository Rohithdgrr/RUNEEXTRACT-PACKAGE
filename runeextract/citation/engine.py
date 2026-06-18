"""Citation engine — auto-cite claims with source sentence markers."""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from runeextract.citation.sentence import split_sentences, tokenize_words
from runeextract.models.document import Document
from runeextract.rag.types import Citation

logger = logging.getLogger(__name__)


@dataclass
class CitationResult:
    """Result of citing one or more claims against a source document."""
    citations: List[Citation] = field(default_factory=list)
    coverage: float = 0.0
    source_sentences: List[str] = field(default_factory=list)


def _word_overlap(query_words: set, sentence_words: set) -> float:
    """Jaccard-like word overlap between two word sets."""
    if not query_words or not sentence_words:
        return 0.0
    intersection = query_words & sentence_words
    union = query_words | sentence_words
    return len(intersection) / max(len(union), 1)


def _sentences_from_source(source: Union[str, Document]) -> Tuple[List[str], str]:
    """Normalise source to (sentences list, full text)."""
    if isinstance(source, Document):
        text = source.text
    else:
        text = str(source)
    sents = split_sentences(text)
    return sents, text


def _embed_similarity(
    embed_fn: Callable[[List[str]], List[List[float]]],
    sentences: List[str],
    claim: str,
) -> List[float]:
    """Score each sentence by embedding similarity to the claim."""
    all_texts = [claim] + sentences
    vectors = embed_fn(all_texts)
    if not vectors or len(vectors) < 2:
        return [0.0] * len(sentences)
    claim_vec = vectors[0]
    sent_vecs = vectors[1:]
    scores = []
    for sv in sent_vecs:
        dot = sum(a * b for a, b in zip(claim_vec, sv))
        n1 = sum(a * a for a in claim_vec) ** 0.5
        n2 = sum(b * b for b in sv) ** 0.5
        denom = max(n1 * n2, 1e-12)
        scores.append(dot / denom)
    return scores


class CitationEngine:
    """Auto-cite claims by matching them to source sentences.

    Supports three matching strategies:
        - ``"overlap"`` — word-level Jaccard overlap (fast, no model needed)
        - ``"embedding"`` — cosine similarity via an embedding function
        - ``"hybrid"`` — weighted combination of overlap + embedding

    Usage::

        from runeextract.citation import CitationEngine
        from runeextract.models.document import Document

        doc = Document(text="Paris is the capital of France. It has the Eiffel Tower.")
        engine = CitationEngine(doc)
        result = engine.cite(["Paris is the capital of France."])
        # result.citations[0].text == "Paris is the capital of France."
        # result.citations[0].page is None  (source not page-structured)
    """

    def __init__(
        self,
        source: Union[str, Document],
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        strategy: str = "overlap",
        top_k: int = 3,
        min_score: float = 0.1,
        hybrid_weight: float = 0.5,
    ):
        self._source = source
        self._sentences, self._full_text = _sentences_from_source(source)
        self._sentence_words = [set(tokenize_words(s)) for s in self._sentences]
        self._embed_fn = embed_fn
        self.strategy = strategy
        self.top_k = top_k
        self.min_score = min_score
        self.hybrid_weight = hybrid_weight

    def cite(self, claims: Union[str, List[str]]) -> CitationResult:
        """Cite one or more claims against the source document.

        Args:
            claims: A single claim string or list of claim strings.

        Returns:
            A ``CitationResult`` with matched citations and metadata.
        """
        if isinstance(claims, str):
            claims = [claims]
        if not claims:
            return CitationResult()

        all_citations: List[Citation] = []
        matched_sentences: set = set()

        for claim in claims:
            citations = self._cite_single(claim)
            all_citations.extend(citations)
            for c in citations:
                matched_sentences.add(c.text)

        total_sents = len(self._sentences) or 1
        coverage = len(matched_sentences) / total_sents

        return CitationResult(
            citations=all_citations,
            coverage=coverage,
            source_sentences=self._sentences,
        )

    def _cite_single(self, claim: str) -> List[Citation]:
        """Find the best-matching source sentences for a single claim."""
        claim_words = set(tokenize_words(claim))

        if self.strategy == "overlap":
            scores = self._score_overlap(claim_words)
        elif self.strategy == "embedding" and self._embed_fn is not None:
            scores = _embed_similarity(self._embed_fn, self._sentences, claim)
        elif self.strategy == "hybrid" and self._embed_fn is not None:
            overlap_scores = self._score_overlap(claim_words)
            embed_scores = _embed_similarity(self._embed_fn, self._sentences, claim)
            w = self.hybrid_weight
            scores = [
                w * o + (1 - w) * e
                for o, e in zip(overlap_scores, embed_scores)
            ]
        else:
            scores = self._score_overlap(claim_words)

        ranked = sorted(
            [
                (score, i, self._sentences[i])
                for i, score in enumerate(scores)
                if score >= self.min_score
            ],
            key=lambda x: -x[0],
        )

        citations: List[Citation] = []
        for score, idx, sent_text in ranked[: self.top_k]:
            citations.append(
                Citation(
                    text=sent_text,
                    source=getattr(self._source, "source_path", "") or str(self._source),
                    page=getattr(self._source, "metadata", {}).get("page"),
                    chunk_index=idx,
                    relevance_score=score,
                )
            )
        return citations

    def _score_overlap(self, claim_words: set) -> List[float]:
        """Score each source sentence by word overlap with the claim."""
        return [
            _word_overlap(claim_words, sw) for sw in self._sentence_words
        ]


def cite_document(
    source: Union[str, Document],
    claims: Union[str, List[str]],
    embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    strategy: str = "overlap",
    top_k: int = 3,
    min_score: float = 0.1,
) -> CitationResult:
    """Convenience function for one-off citation extraction.

    Usage::

        from runeextract.citation import cite_document

        result = cite_document("Paris is the capital.", "Paris is in France.")
        # result.citations[0].text == "Paris is the capital."
    """
    engine = CitationEngine(
        source=source,
        embed_fn=embed_fn,
        strategy=strategy,
        top_k=top_k,
        min_score=min_score,
    )
    return engine.cite(claims)
