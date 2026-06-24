"""
Query intent classification and decomposition for RAG pipelines.

``QueryRouter`` analyses a natural-language question to determine its
intent (factual, analytical, comparative, summarization, exploratory)
and optionally decomposes it into sub-queries for multi-step retrieval.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    SUMMARIZATION = "summarization"
    EXPLORATORY = "exploratory"


@dataclass
class DecomposedQuery:
    """Result of query decomposition."""
    original: str
    intent: QueryIntent
    sub_queries: List[str] = field(default_factory=list)
    metadata_filter: Dict[str, str] = field(default_factory=dict)
    confidence: float = 1.0


_INTENT_PATTERNS: List[Tuple[QueryIntent, List[str]]] = [
    (QueryIntent.COMPARATIVE, [
        r"\b(compare|contrast|difference|similar|versus|vs\.?|advantage)\b",
        r"\b(better|worse|pros?|cons?)\b",
        r"(how does .+ differ|what is the difference)",
    ]),
    (QueryIntent.ANALYTICAL, [
        r"\b(why|how does|how can|explain|reason|cause|impact|affect|influence)\b",
        r"\b(analyze|analysis|trend|correlation|relation)\b",
        r"(what caused|what led to)",
    ]),
    (QueryIntent.SUMMARIZATION, [
        r"\b(summarize|summary|overview|key.?points|main.?points|recap|brief)\b",
        r"(give me the gist|tl;?dr|in short)",
    ]),
    (QueryIntent.EXPLORATORY, [
        r"\b(enumerate|list all|find all|search for|explore|discover)\b",
        r"\b(types?|kinds?|categories|examples?|instances)\b",
    ]),
]

_FILTER_PATTERNS = [
    (r"(?:in|from|during)\s+(\d{4})\s*(?:to|through|-)\s*(\d{4})", "year_range"),
    (r"(?:in|from|during)\s+(\d{4})", "year"),
    (r"(?:author|by)\s+[\"'](.+?)[\"']", "author"),
    (r"(?:section|chapter|article)\s+(\d+)", "section"),
]


class QueryRouter:
    """Classify and optionally decompose a RAG query.

    Works with or without an LLM — rule-based classification and
    keyword-based filter extraction are always available.
    """

    def __init__(self, llm_complete: Optional[callable] = None):
        self._llm = llm_complete

    def classify(self, query: str) -> QueryIntent:
        """Determine the intent of a query using pattern matching (no LLM).

        Returns the most specific matching intent.
        """
        lower = query.lower()
        for intent, patterns in _INTENT_PATTERNS:
            for pat in patterns:
                if re.search(pat, lower):
                    logger.debug("Classified as %s via pattern %s", intent.value, pat)
                    return intent
        return QueryIntent.FACTUAL

    def extract_filters(self, query: str) -> Dict[str, str]:
        """Extract metadata filters from a query using regex patterns."""
        filters: Dict[str, str] = {}
        for pat, key in _FILTER_PATTERNS:
            m = re.search(pat, query, re.IGNORECASE)
            if m:
                groups = m.groups()
                if len(groups) == 2:
                    filters[key] = f"{groups[0]}-{groups[1]}"
                else:
                    filters[key] = groups[0]
        return filters

    def decompose(self, query: str) -> DecomposedQuery:
        """Decompose a complex query into sub-queries.

        Uses LLM when available, falls back to rule-based splitting on
        conjunctions and punctuation.
        """
        intent = self.classify(query)
        filters = self.extract_filters(query)
        sub_queries = []

        if self._llm:
            try:
                prompt = (
                    "Decompose the following question into 1-3 simpler "
                    "sub-questions that together would fully answer it. "
                    "Return one sub-question per line, no numbering.\n\n"
                    f"Question: {query}"
                )
                response = self._llm(prompt, max_tokens=200)
                sub_queries = [
                    line.strip("- ").strip()
                    for line in response.strip().split("\n")
                    if line.strip()
                ]
            except Exception as exc:
                logger.debug("LLM decomposition failed: %s", exc)

        if not sub_queries:
            sub_queries = self._rule_based_split(query)

        return DecomposedQuery(
            original=query,
            intent=intent,
            sub_queries=sub_queries if len(sub_queries) > 1 else [query],
            metadata_filter=filters,
        )

    @staticmethod
    def _rule_based_split(query: str) -> List[str]:
        """Split on coordinating conjunctions and question marks."""
        parts = re.split(r"\s+(and|or|but|,|\?)\s+", query)
        parts = [p.strip().strip("?").strip() for p in parts if p.strip()]
        return [p for p in parts if len(p) > 10]
