"""
Tests for core data models.
"""

import pytest
from runeextract.models.document import Document, Table, Image, Chunk, ChunkingStrategy


def test_document_creation():
    """Test basic document creation."""
    doc = Document(
        text="Sample text",
        source_type="pdf",
        source_path="test.pdf"
    )
    
    assert doc.text == "Sample text"
    assert doc.source_type == "pdf"
    assert doc.source_path == "test.pdf"
    assert doc.tables == []
    assert doc.images == []
    assert doc.metadata == {}


def test_document_with_tables():
    """Test document with tables."""
    table = Table(
        rows=[["A", "B"], ["1", "2"]],
        columns=["Col1", "Col2"]
    )
    
    doc = Document(
        text="Sample text",
        tables=[table],
        source_type="pdf"
    )
    
    assert len(doc.tables) == 1
    assert doc.tables[0].columns == ["Col1", "Col2"]


def test_document_with_images():
    """Test document with images."""
    image = Image(
        data=b"fake_image_data",
        format="png"
    )
    
    doc = Document(
        text="Sample text",
        images=[image],
        source_type="pdf"
    )
    
    assert len(doc.images) == 1
    assert doc.images[0].format == "png"


def test_chunking_fixed_size():
    """Test fixed-size chunking."""
    doc = Document(
        text="A" * 1000 + "B" * 1000,
        source_type="pdf"
    )
    
    chunks = doc.chunks(strategy=ChunkingStrategy.FIXED_SIZE, size=500, overlap=50)
    
    assert len(chunks) > 1
    assert all(isinstance(chunk, Chunk) for chunk in chunks)


def test_chunking_semantic():
    """Test semantic chunking."""
    doc = Document(
        text="Paragraph 1\n\nParagraph 2\n\nParagraph 3",
        source_type="pdf"
    )
    
    chunks = doc.chunks(strategy=ChunkingStrategy.SEMANTIC, size=100)
    
    assert len(chunks) >= 1
    assert all(isinstance(chunk, Chunk) for chunk in chunks)


def test_table_to_dataframe():
    """Test table to DataFrame conversion."""
    table = Table(
        rows=[["1", "2"], ["3", "4"]],
        columns=["A", "B"]
    )
    
    df = table.to_dataframe()
    
    assert df is not None
    assert len(df) == 2
    assert list(df.columns) == ["A", "B"]


def test_document_to_dict():
    """Test document serialization to dict."""
    table = Table(
        rows=[["1", "2"]],
        columns=["A", "B"]
    )
    
    doc = Document(
        text="Sample text",
        tables=[table],
        source_type="pdf",
        metadata={"title": "Test"}
    )
    
    doc_dict = doc.to_dict()
    
    assert doc_dict["text"] == "Sample text"
    assert doc_dict["source_type"] == "pdf"
    assert len(doc_dict["tables"]) == 1
    assert doc_dict["metadata"]["title"] == "Test"
