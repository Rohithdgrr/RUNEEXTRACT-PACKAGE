"""
Citation module — auto-cite claims with source sentence markers.
"""

from runeextract.citation.engine import CitationEngine, CitationResult, cite_document
from runeextract.citation.sentence import split_sentences, tokenize_words

__all__ = [
    "CitationEngine", "CitationResult", "cite_document",
    "split_sentences", "tokenize_words",
]
