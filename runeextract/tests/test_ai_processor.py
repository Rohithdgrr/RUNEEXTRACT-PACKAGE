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


@patch("runeextract.processors.ai.AIProcessor._call")
def test_embed_single_string(mock_call):
    """embed with single string returns list of one vector."""
    from runeextract.processors.providers.openai_compat import embed as _embed_fn
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value.data = [
        MagicMock(embedding=[0.1, 0.2, 0.3])
    ]
    ai = AIProcessor(provider="openai", api_key="sk-test")
    ai._client = mock_client
    result = ai.embed("hello")
    assert len(result) == 1
    assert len(result[0]) == 3


@patch("runeextract.processors.ai.AIProcessor._call")
def test_embed_multiple_texts(mock_call):
    """embed with list of strings returns one vector per string."""
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value.data = [
        MagicMock(embedding=[0.1, 0.2]),
        MagicMock(embedding=[0.3, 0.4]),
    ]
    ai = AIProcessor(provider="openai", api_key="sk-test")
    ai._client = mock_client
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


@patch("runeextract.processors.ai.AIProcessor._call")
def test_extract_entities_structured(mock_call):
    """extract_entities requests JSON response format."""
    mock_call.return_value = '[{"type": "person", "name": "Alice"}]'
    ai = AIProcessor(provider="openai", api_key="sk-test")
    entities = ai.extract_entities("Alice went to Paris.")
    assert len(entities) == 1
    assert entities[0]["type"] == "person"
    assert entities[0]["name"] == "Alice"


@patch("runeextract.processors.ai.AIProcessor._call")
def test_generate_flashcards_structured(mock_call):
    """generate_flashcards requests JSON response format."""
    mock_call.return_value = '{"flashcards": [{"front": "Q", "back": "A"}]}'
    ai = AIProcessor(provider="openai", api_key="sk-test")
    cards = ai.generate_flashcards("Some text.", n=1)
    assert len(cards) == 1
    assert cards[0]["front"] == "Q"
    assert cards[0]["back"] == "A"


def test_document_ask_stream():
    """Document.ask_stream yields tokens from _call_stream."""
    from runeextract.models.document import Document, Chunk, ChunkingStrategy
    from runeextract.processors.ai import AIProcessor
    ai = AIProcessor(provider="ollama", api_key="")
    ai._call_stream = MagicMock(return_value=iter(["Hello", " ", "World"]))
    doc = Document(text="Test content", source_type="text")
    doc._chunks = [Chunk(text="Test content", chunk_id="chunk_0", start_index=0, end_index=12)]
    doc.retrieve = MagicMock(return_value=[
        (doc._chunks[0], 0.95)
    ])
    tokens = list(doc.ask_stream("What is this?", top_k=1, ai_processor=ai))
    assert "".join(tokens) == "Hello World"
    ai._call_stream.assert_called_once()


def test_document_ask_stream_no_content():
    """Document.ask_stream yields fallback when no chunks match."""
    from runeextract.models.document import Document, Chunk, ChunkingStrategy
    doc = Document(text="", source_type="text")
    tokens = list(doc.ask_stream("What is this?"))
    assert "".join(tokens) == "No content available to answer the question."


def test_chat_session_ask_stream():
    """ChatSession.ask_stream yields tokens and stores assistant message."""
    from runeextract.models.document import Document, Chunk, ChunkingStrategy, ChatSession
    from runeextract.processors.ai import AIProcessor
    ai = AIProcessor(provider="ollama", api_key="")
    ai._call_stream = MagicMock(return_value=iter(["Hello", " ", "World"]))
    doc = Document(text="Test content", source_type="text")
    doc._chunks = [Chunk(text="Test content", chunk_id="chunk_0", start_index=0, end_index=12)]
    doc.retrieve = MagicMock(return_value=[(doc._chunks[0], 0.95)])
    chat = ChatSession(document=doc, ai_processor=ai)
    tokens = list(chat.ask_stream("What is this?", top_k=1))
    assert "".join(tokens) == "Hello World"
    assert len(chat.messages) == 2  # user + assistant
    assert chat.messages[0]["role"] == "user"
    ai._call_stream.assert_called_once()


def test_chat_session_ask():
    """ChatSession.ask stores conversation history across calls."""
    from runeextract.models.document import Document, Chunk, ChunkingStrategy, ChatSession
    from runeextract.processors.ai import AIProcessor
    ai = AIProcessor(provider="ollama", api_key="")
    ai._call = MagicMock(side_effect=["First answer.", "Second answer."])
    doc = Document(text="Some document text here.", source_type="text")
    doc._chunks = [Chunk(text="Some document text here.", chunk_id="chunk_0", start_index=0, end_index=26)]
    doc.retrieve = MagicMock(return_value=[(doc._chunks[0], 0.95)])
    chat = ChatSession(document=doc, ai_processor=ai)
    answer1 = chat.ask("First question?", top_k=1)
    assert answer1 == "First answer."
    assert len(chat.messages) == 2  # user + assistant
    answer2 = chat.ask("Second question?", top_k=1)
    assert answer2 == "Second answer."
    assert len(chat.messages) == 4  # user, assistant, user, assistant
    assert ai._call.call_count == 2


def test_chat_session_no_document():
    """ChatSession works without a Document (general conversation)."""
    from runeextract.models.document import ChatSession
    from runeextract.processors.ai import AIProcessor
    ai = AIProcessor(provider="ollama", api_key="")
    ai._call = MagicMock(return_value="General answer.")
    chat = ChatSession(ai_processor=ai)
    answer = chat.ask("Hello?", top_k=1)
    assert answer == "General answer."
    assert len(chat.messages) == 2


def test_chat_session_stream_no_document():
    """ChatSession.ask_stream works without a Document."""
    from runeextract.models.document import ChatSession
    from runeextract.processors.ai import AIProcessor
    ai = AIProcessor(provider="ollama", api_key="")
    ai._call_stream = MagicMock(return_value=iter(["streamed"]))
    chat = ChatSession(ai_processor=ai)
    tokens = list(chat.ask_stream("Hi?"))
    assert "".join(tokens) == "streamed"
    assert len(chat.messages) == 2


def test_chat_session_system_prompt():
    """ChatSession accepts a custom system prompt."""
    from runeextract.models.document import ChatSession
    chat = ChatSession(system_prompt="You are a math tutor.")
    assert len(chat.messages) == 1
    assert chat.messages[0]["role"] == "system"
    assert chat.messages[0]["content"] == "You are a math tutor."


def test_chat_session_manual_messages():
    """ChatSession.add_user_message/add_assistant_message work correctly."""
    from runeextract.models.document import ChatSession
    chat = ChatSession()
    chat.add_user_message("Q1")
    chat.add_assistant_message("A1")
    chat.add_user_message("Q2")
    assert len(chat.messages) == 3
    assert chat.messages[0]["role"] == "user"
    assert chat.messages[0]["content"] == "Q1"
    assert chat.messages[1]["role"] == "assistant"
    assert chat.messages[1]["content"] == "A1"
