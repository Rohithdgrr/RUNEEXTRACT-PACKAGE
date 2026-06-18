"""Sentence-level text utilities for citation and evidence extraction."""

import re
from typing import List


_SENTENCE_BOUNDARY = re.compile(
    r'(?<!\b\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=[.?!])\s+(?=[A-Z"\'({])'
)


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using a heuristic regex.

    Handles common abbreviations (Mr., Dr., U.S., etc.) by
    negative lookbehind for single-letter/two-letter patterns.
    """
    sentences = _SENTENCE_BOUNDARY.split(text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize_words(text: str) -> List[str]:
    """Lowercase tokenization for overlap scoring."""
    return re.findall(r"[a-z0-9]+", text.lower())
