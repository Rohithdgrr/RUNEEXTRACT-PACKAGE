"""Tests for AI processor with multi-provider, embeddings, and Q&A."""

import os
from unittest.mock import patch, MagicMock

import pytest
from runeextract.processors.ai import AIProcessor
from runeextract.exceptions import ExtractionError, DependencyMissingError


def test_no_api_key_raises():
    """AIProcessor raises ExtractionError when no API key is set."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ExtractionError, match="API key not set"):
            AIProcessor(provider="openai", api_key=None)


def test_unknown_provider_raises():
    """AIProcessor raises ExtractionError for unknown provider."""
    with pytest.raises(ExtractionError, match="Unknown provider"):
        AIProcessor(provider="unknown", api_key="sk-test")


def test_ollama_no_key_required():
    """Ollama provider does not require an API key."""
    ai = AIProcessor(provider="ollama", api_key="")
    assert ai.provider == "ollama"


def test_provider_enum_openai():
    ai = AIProcessor(provider="openai", api_key="sk-test")
    assert ai.provider == "openai"


def test_provider_enum_anthropic():
    ai = AIProcessor(provider="anthropic", api_key="sk-ant-test")
    assert ai.provider == "anthropic"


def test_provider_enum_gemini():
    ai = AIProcessor(provider="gemini", api_key="gem-test")
    assert ai.provider == "gemini"


def test_provider_case_insensitive():
    ai = AIProcessor(provider="OpenAI", api_key="sk-test")
    assert ai.provider == "openai"


@patch("runeextract.processors.ai.AIProcessor._call")
def test_answer_question(mock_call):
    """answer_question delegates to _call with correct prompt."""
    mock_call.return_value = "Paris is the capital of France."
    ai = AIProcessor(provider="openai", api_key="sk-test")
    answer = ai.answer_question("What is the capital of France?", "France is a country. Paris is its capital.")
    assert answer == "Paris is the capital of France."
    mock_call.assert_called_once()
    args, kwargs = mock_call.call_args
    assert "Question: What is the capital of France?" in kwargs.get("user", args[1])
    assert "Context:" in kwargs.get("user", args[1])


@patch("runeextract.processors.ai.AIProcessor._embed_openai")
def test_embed_single_string(mock_embed_openai):
    """embed with single string returns list of one vector."""
    mock_embed_openai.return_value = [[0.1, 0.2, 0.3]]
    ai = AIProcessor(provider="openai", api_key="sk-test")
    result = ai.embed("hello")
    assert len(result) == 1
    assert len(result[0]) == 3


@patch("runeextract.processors.ai.AIProcessor._embed_openai")
def test_embed_multiple_texts(mock_embed_openai):
    """embed with list of strings returns one vector per string."""
    mock_embed_openai.return_value = [[0.1, 0.2], [0.3, 0.4]]
    ai = AIProcessor(provider="openai", api_key="sk-test")
    result = ai.embed(["hello", "world"])
    assert len(result) == 2


@patch("runeextract.processors.ai.AIProcessor._call")
def test_summarize(mock_call):
    """summarize delegates to _call."""
    mock_call.return_value = "Summary text."
    ai = AIProcessor(provider="openai", api_key="sk-test")
    result = ai.summarize("Long text here. " * 100, max_words=50)
    assert result == "Summary text."


def test_embed_unsupported_provider():
    """embed raises if provider does not support embeddings."""
    ai = AIProcessor(provider="anthropic", api_key="sk-ant-test")
    with pytest.raises(ExtractionError, match="Embedding not supported"):
        ai.embed("test")


@patch("runeextract.processors.ai.AIProcessor._call_openai")
def test_extract_entities_structured(mock_call_openai):
    """extract_entities requests JSON response format."""
    mock_call_openai.return_value = '[{"type": "person", "name": "Alice"}]'
    ai = AIProcessor(provider="openai", api_key="sk-test")
    entities = ai.extract_entities("Alice went to Paris.")
    assert len(entities) == 1
    assert entities[0]["type"] == "person"
    assert entities[0]["name"] == "Alice"


@patch("runeextract.processors.ai.AIProcessor._call_openai")
def test_generate_flashcards_structured(mock_call_openai):
    """generate_flashcards requests JSON response format."""
    mock_call_openai.return_value = '{"flashcards": [{"front": "Q", "back": "A"}]}'
    ai = AIProcessor(provider="openai", api_key="sk-test")
    cards = ai.generate_flashcards("Some text.", n=1)
    assert len(cards) == 1
    assert cards[0]["front"] == "Q"
    assert cards[0]["back"] == "A"
