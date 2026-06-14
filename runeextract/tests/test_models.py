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


def test_document_to_json():
    """Test document serialization to JSON."""
    doc = Document(text="Hello", source_type="md")
    json_str = doc.to_json()
    import json
    parsed = json.loads(json_str)
    assert parsed["text"] == "Hello"
    assert parsed["source_type"] == "md"


def test_document_to_markdown():
    """Test document to Markdown conversion."""
    table = Table(rows=[["1", "2"]], columns=["A", "B"])
    doc = Document(text="Some text.", tables=[table], source_type="md",
                   metadata={"title": "MyDoc"})
    md = doc.to_markdown()
    assert "# MyDoc" in md
    assert "Some text" in md
    assert "| A | B |" in md


def test_chunking_by_page():
    """Test page-based chunking."""
    doc = Document(
        text="Page1\nPage2\nPage3",
        metadata={"page_breaks": [6, 12]},
        source_type="pdf"
    )
    chunks = doc.chunks(strategy=ChunkingStrategy.BY_PAGE)
    assert len(chunks) == 3


def test_chunking_by_heading():
    """Test heading-based chunking."""
    doc = Document(
        text="# Intro\nText\n## Details\nMore text\n### End\nFinal",
        source_type="md"
    )
    chunks = doc.chunks(strategy=ChunkingStrategy.BY_HEADING)
    assert len(chunks) >= 1


def test_chunking_by_heading_no_heading():
    """Test heading chunking falls back to fixed_size when no headings."""
    doc = Document(
        text="Plain text without any headings whatsoever",
        source_type="txt"
    )
    chunks = doc.chunks(strategy=ChunkingStrategy.BY_HEADING)
    assert len(chunks) >= 1


def test_chunking_strategy_enum():
    """Test chunking strategy enum values."""
    assert ChunkingStrategy.BY_PAGE.value == "by_page"
    assert ChunkingStrategy.BY_HEADING.value == "by_heading"
    assert ChunkingStrategy.SEMANTIC.value == "semantic"
    assert ChunkingStrategy.FIXED_SIZE.value == "fixed_size"
    assert ChunkingStrategy.BY_TOKEN.value == "by_token"


def test_token_count_estimate():
    """Test token count falls back to character estimate when tiktoken not available."""
    doc = Document(text="Hello world!", source_type="txt")
    count = doc.token_count()
    assert count >= 1
    assert isinstance(count, int)


def test_chunk_token_count():
    """Test Chunk token_count method."""
    chunk = Chunk(text="Hello world!", chunk_id="c0", start_index=0, end_index=12)
    count = chunk.token_count()
    assert count >= 1
    assert isinstance(count, int)


def test_chunking_by_token():
    """Test token-based chunking."""
    text = "word " * 2000
    doc = Document(text=text, source_type="txt")
    chunks = doc.chunks(strategy=ChunkingStrategy.BY_TOKEN, size=100, overlap=10)
    assert len(chunks) > 1
    assert all(isinstance(c, Chunk) for c in chunks)


def test_to_dict_includes_token_count():
    """Test that to_dict includes token_count in chunks."""
    doc = Document(text="Hello world", source_type="txt")
    doc.chunks(strategy=ChunkingStrategy.FIXED_SIZE, size=100, overlap=0)
    d = doc.to_dict()
    for chunk_dict in d["chunks"]:
        assert "token_count" in chunk_dict
        assert isinstance(chunk_dict["token_count"], int)


def test_document_id_auto_generated():
    """Document gets a unique document_id on creation."""
    doc1 = Document(text="a", source_type="txt")
    doc2 = Document(text="b", source_type="txt")
    assert doc1.document_id
    assert doc2.document_id
    assert doc1.document_id != doc2.document_id


def test_chunk_parent_document_id():
    """Chunks reference their parent document after chunking."""
    doc = Document(text="Hello world", source_type="txt")
    chunks = doc.chunks(strategy=ChunkingStrategy.FIXED_SIZE, size=5, overlap=0)
    for chunk in chunks:
        assert chunk.parent_document_id == doc.document_id


def test_word_count():
    doc = Document(text="one two three four", source_type="txt")
    assert doc.word_count == 4


def test_text_length():
    doc = Document(text="hello", source_type="txt")
    assert doc.text_length == 5


def test_merge_documents():
    doc1 = Document(text="First", source_type="md", source_path="a.md")
    doc2 = Document(text="Second", source_type="md", source_path="b.md")
    merged = Document.merge([doc1, doc2])
    assert merged.text == "First\n\nSecond"
    assert merged.source_type == "merged"
    assert "a.md" in merged.source_path
    assert "b.md" in merged.source_path


def test_merge_single():
    doc = Document(text="Alone", source_type="txt")
    merged = Document.merge([doc])
    assert merged.text == "Alone"


def test_merge_empty():
    merged = Document.merge([])
    assert merged.text == ""


def test_to_openai_messages_default():
    doc = Document(text="Hello world", source_type="txt")
    msgs = doc.to_openai_messages()
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "Hello world"


def test_to_openai_messages_with_system():
    doc = Document(text="Hello world", source_type="txt")
    msgs = doc.to_openai_messages(system_message="Be helpful")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "Be helpful"
    assert msgs[1]["role"] == "user"


def test_to_openai_messages_with_chunks():
    doc = Document(text="A" * 500 + "B" * 500, source_type="txt")
    doc.chunks(strategy=ChunkingStrategy.FIXED_SIZE, size=100, overlap=0)
    msgs = doc.to_openai_messages()
    assert len(msgs) > 1
    for msg in msgs:
        assert msg["role"] == "user"
        assert len(msg["content"]) <= 100
