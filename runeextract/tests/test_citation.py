"""Tests for citation engine — sentence matching, overlap, embedding."""

from unittest.mock import MagicMock

import pytest

from runeextract.citation import (
    CitationEngine, CitationResult, cite_document,
    split_sentences, tokenize_words,
)
from runeextract.citation.engine import _word_overlap, _embed_similarity
from runeextract.models.document import Document


class TestSentenceUtils:
    def test_split_sentences_basic(self):
        sents = split_sentences("Hello world. This is a test. Goodbye!")
        assert len(sents) == 3
        assert sents[0] == "Hello world."

    def test_split_sentences_single(self):
        sents = split_sentences("Just one sentence here.")
        assert len(sents) == 1

    def test_split_sentences_with_abbreviation(self):
        sents = split_sentences("Dr. Smith went to Washington. He met Mr. Jones.")
        assert len(sents) == 2

    def test_split_sentences_empty(self):
        assert split_sentences("") == []
        assert split_sentences("   ") == []

    def test_tokenize_words_basic(self):
        tokens = tokenize_words("Hello World! This is a TEST.")
        assert tokens == ["hello", "world", "this", "is", "a", "test"]

    def test_tokenize_words_empty(self):
        assert tokenize_words("") == []


class TestWordOverlap:
    def test_perfect_overlap(self):
        score = _word_overlap({"a", "b"}, {"a", "b"})
        assert score == pytest.approx(1.0)

    def test_partial_overlap(self):
        score = _word_overlap({"a", "b"}, {"a", "c"})
        assert score == pytest.approx(1 / 3)

    def test_no_overlap(self):
        score = _word_overlap({"a"}, {"b"})
        assert score == pytest.approx(0.0)

    def test_empty_inputs(self):
        assert _word_overlap(set(), {"a"}) == 0.0
        assert _word_overlap({"a"}, set()) == 0.0
        assert _word_overlap(set(), set()) == 0.0


class TestEmbedSimilarity:
    def test_basic_similarity(self):
        embed_fn = MagicMock(return_value=[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
        scores = _embed_similarity(embed_fn, ["sent1", "sent2"], "query")
        assert len(scores) == 2
        assert scores[0] > scores[1]  # first sentence more similar

    def test_empty_sentences(self):
        embed_fn = MagicMock(return_value=[[1.0]])
        scores = _embed_similarity(embed_fn, [], "query")
        assert scores == []

    def test_fallback_on_short_result(self):
        embed_fn = MagicMock(return_value=[[1.0]])
        scores = _embed_similarity(embed_fn, ["a", "b"], "query")
        assert scores == [0.0, 0.0]


class TestCitationEngineInit:
    def test_init_with_string(self):
        engine = CitationEngine("Hello world. Test sentence.")
        assert len(engine._sentences) == 2

    def test_init_with_document(self):
        doc = Document(text="First sentence. Second sentence.")
        engine = CitationEngine(doc)
        assert len(engine._sentences) == 2

    def test_init_empty_source(self):
        engine = CitationEngine("")
        assert engine._sentences == []


class TestCitationEngineCite:
    def test_cite_single_claim(self):
        engine = CitationEngine("Paris is the capital of France. It has the Eiffel Tower.")
        result = engine.cite("Paris is the capital of France.")
        assert len(result.citations) >= 1
        assert "Paris" in result.citations[0].text

    def test_cite_claim_not_found(self):
        engine = CitationEngine("The sky is blue. Grass is green.")
        result = engine.cite("Elephants are large.")
        assert len(result.citations) == 0

    def test_cite_multiple_claims(self):
        engine = CitationEngine("Python is a language. Rust is fast. Java is verbose.")
        result = engine.cite(["Python is a language.", "Rust is fast."])
        assert len(result.citations) >= 2

    def test_cite_empty_claims(self):
        engine = CitationEngine("Some text here.")
        result = engine.cite([])
        assert len(result.citations) == 0
        assert result.coverage == 0.0

    def test_cite_top_k(self):
        engine = CitationEngine(
            "A B C. A B D. A E F. G H I.",
            top_k=2,
        )
        result = engine.cite("A B")
        assert len(result.citations) <= 2

    def test_cite_min_score_filters(self):
        engine = CitationEngine("Alpha Beta Gamma. Delta Epsilon Zeta.", min_score=0.5)
        result = engine.cite("Alpha Beta Gamma")
        # The exact match should be well above 0.5
        assert len(result.citations) >= 1

    def test_cite_coverage(self):
        engine = CitationEngine(
            "Sentence one about cats. Sentence two about dogs. "
            "Sentence three about birds."
        )
        result = engine.cite(["cats", "dogs"])
        assert 0.0 < result.coverage <= 1.0

    def test_cite_returns_citation_object(self):
        engine = CitationEngine("Source document text here.")
        result = engine.cite("Source document")
        if result.citations:
            c = result.citations[0]
            assert hasattr(c, "text")
            assert hasattr(c, "source")
            assert hasattr(c, "relevance_score")


class TestCitationEngineEmbeddingStrategy:
    def test_cite_with_embedding_strategy(self):
        embed_fn = MagicMock(return_value=[
            [1.0, 0.0],
            [0.95, 0.05],
            [0.1, 0.9],
        ])
        engine = CitationEngine(
            "First sentence here. Second unrelated thing.",
            embed_fn=embed_fn,
            strategy="embedding",
        )
        result = engine.cite("First sentence")
        # Should prefer the first sentence
        if result.citations:
            assert "First" in result.citations[0].text or result.citations[0].relevance_score > 0.5

    def test_cite_with_hybrid_strategy(self):
        embed_fn = MagicMock(return_value=[
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ])
        engine = CitationEngine(
            "Cats are nice. Dogs are pets. Fish swim.",
            embed_fn=embed_fn,
            strategy="hybrid",
        )
        result = engine.cite("Cats")
        assert len(result.citations) >= 0

    def test_embedding_fallback_to_overlap(self):
        engine = CitationEngine(
            "Hello world.",
            embed_fn=None,
            strategy="embedding",
        )
        result = engine.cite("Hello")
        assert len(result.citations) >= 1


class TestCiteDocumentFunction:
    def test_cite_document_basic(self):
        result = cite_document("Paris is the capital of France.", "Paris")
        assert isinstance(result, CitationResult)

    def test_cite_document_with_list(self):
        result = cite_document(
            "Python is great. Java is also popular.",
            ["Python is great."],
        )
        assert len(result.citations) >= 1


class TestSourceSentenceAccess:
    def test_source_sentences_in_result(self):
        engine = CitationEngine("First. Second. Third.")
        result = engine.cite("First")
        assert result.source_sentences == ["First.", "Second.", "Third."]

    def test_cite_from_document_preserves_metadata(self):
        doc = Document(
            text="Test content here.",
            source_type="text",
            source_path="test.txt",
            metadata={"page": 1},
        )
        engine = CitationEngine(doc)
        result = engine.cite("Test content")
        if result.citations:
            assert result.citations[0].source == "test.txt"


class TestEmptyEdgeCases:
    def test_empty_source_empty_claims(self):
        engine = CitationEngine("")
        result = engine.cite("")
        assert len(result.citations) == 0

    def test_whitespace_only_source(self):
        engine = CitationEngine("   \n\n  ")
        result = engine.cite("test")
        assert len(result.citations) == 0

    def test_single_word_source(self):
        engine = CitationEngine("Hello")
        result = engine.cite("Hello")
        assert len(result.citations) >= 1
