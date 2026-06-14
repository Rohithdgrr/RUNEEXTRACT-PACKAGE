"""Tests for in-memory extraction, cache integration, and vector stores."""

import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

import pytest
from runeextract import extract_from_bytes, extract_from_string, extract
from runeextract.exceptions import UnsupportedFormatError
from runeextract.models.document import Document


def test_extract_from_string_markdown():
    """Extract markdown from a string."""
    md = "# Hello\n\nThis is **bold** text."
    doc = extract_from_string(md, "test.md")
    assert doc.source_type == "markdown"
    assert "Hello" in doc.text


def test_extract_from_string_html():
    """Extract HTML from a string."""
    html = "<html><body><h1>Title</h1><p>Paragraph</p></body></html>"
    doc = extract_from_string(html, "page.html")
    assert "Title" in doc.text
    assert "Paragraph" in doc.text


def test_extract_from_bytes_markdown():
    """Extract markdown from bytes."""
    md = b"# Hello\n\nThis is **bold** text."
    doc = extract_from_bytes(md, "test.md")
    assert "Hello" in doc.text


def test_extract_from_bytes_unsupported_extension():
    """extract_from_bytes raises UnsupportedFormatError for unknown extension."""
    data = b"some content"
    with pytest.raises(UnsupportedFormatError):
        extract_from_bytes(data, "file.xyz")


def test_extract_from_string_empty():
    """extract_from_string with empty content raises CorruptFileError."""
    from runeextract.exceptions import CorruptFileError
    with pytest.raises(CorruptFileError):
        extract_from_string("", "empty.md")


def test_extract_with_cache_hit():
    """extract with use_cache=True returns cached result on subsequent calls."""
    content = "# Cached Document\nSome text."
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(content)
        path = f.name
    try:
        # First call populates cache
        doc1 = extract(path, use_cache=True)
        assert doc1 is not None

        # Second call should hit cache (same file, same options)
        doc2 = extract(path, use_cache=True)
        assert doc2 is not None
        assert doc2.text == doc1.text
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_extract_with_cache_disabled():
    """Without use_cache, extraction proceeds normally."""
    content = "# No Cache\nText."
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(content)
        path = f.name
    try:
        doc = extract(path, use_cache=False)
        assert doc is not None
        assert "No Cache" in doc.text
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_extract_from_bytes_with_chunking():
    """extract_from_bytes with chunking strategy produces chunks."""
    md = "# Header\n\n" + "Paragraph text. " * 50
    doc = extract_from_bytes(md.encode(), "test.md", chunking_strategy="fixed_size", chunk_size=100)
    chunks = doc.chunks()
    assert len(chunks) >= 1


def test_document_id_in_extracted_doc():
    """Extracted documents have a document_id."""
    doc = extract_from_string("# Hello", "test.md")
    assert doc.document_id
    assert isinstance(doc.document_id, str)


def test_to_chromadb():
    """to_chromadb creates a ChromaDB collection with chunks."""
    chromadb_mock = MagicMock()
    chromadb_settings_mock = MagicMock()
    mock_collection = MagicMock()
    chromadb_mock.PersistentClient.return_value.get_or_create_collection.return_value = mock_collection

    with patch.dict("sys.modules", {"chromadb": chromadb_mock, "chromadb.config": chromadb_settings_mock}):
        doc = Document(text="A" * 2000, source_type="txt")
        doc.chunks(strategy="fixed_size", size=500, overlap=0)
        collection = doc.to_chromadb(collection_name="test_coll", persist_directory="/tmp/chroma_test")
    assert collection is not None
    chromadb_mock.PersistentClient.assert_called_once()
    mock_collection.add.assert_called_once()


def test_to_faiss():
    """to_faiss creates a FAISS index with chunk metadata."""
    faiss_mock = MagicMock()
    np_mock = MagicMock()
    mock_index_instance = MagicMock()
    faiss_mock.IndexFlatL2.return_value = mock_index_instance
    # Mock numpy to avoid C extension reload issues on Python 3.14
    np_mock.float32 = float
    np_mock.random.default_rng.return_value.random.return_value = [[0.1] * 384]

    with patch.dict("sys.modules", {"faiss": faiss_mock, "numpy": np_mock}):
        doc = Document(text="B" * 2000, source_type="txt")
        doc.chunks(strategy="fixed_size", size=500, overlap=0)
        index, meta = doc.to_faiss(index_path="./faiss_test")
    assert index is not None
    assert len(meta) > 1
    assert meta[0]["text"]
    assert meta[0]["document_id"] == doc.document_id
