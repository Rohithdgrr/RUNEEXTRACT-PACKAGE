"""
Integration tests for RuneExtract extractors.

These tests verify the end-to-end functionality of extractors
with actual file content.
"""

import pytest
import tempfile
import os
from pathlib import Path


class TestMarkdownIntegration:
    """Integration tests for Markdown extractor."""
    
    def test_simple_markdown_extraction(self):
        """Test extraction of simple markdown file."""
        from runeextract import extract
        
        # Create a temporary markdown file
        content = """# Test Document

This is a test paragraph.

## Section 1

Some content here.

## Section 2

More content.
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name
        
        try:
            doc = extract(temp_path)
            
            assert doc.text
            assert doc.source_type == "markdown"
            assert "Test Document" in doc.text
            assert "Section 1" in doc.text
        finally:
            os.unlink(temp_path)
    
    def test_markdown_with_table(self):
        """Test extraction of markdown with table."""
        from runeextract import extract
        
        content = """# Document with Table

| Name | Age | City |
|------|-----|------|
| John | 25  | NYC  |
| Jane | 30  | LA   |
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name
        
        try:
            doc = extract(temp_path)
            
            assert len(doc.tables) > 0
            table = doc.tables[0]
            assert "Name" in table.columns
            assert len(table.rows) == 2
        finally:
            os.unlink(temp_path)
    
    def test_markdown_with_frontmatter(self):
        """Test extraction of markdown with YAML frontmatter."""
        from runeextract import extract
        
        content = """---
title: Test Document
author: John Doe
date: 2024-01-01
---

# Content

This is the content.
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name
        
        try:
            doc = extract(temp_path)
            
            assert doc.metadata.get('title') == "Test Document"
            assert doc.metadata.get('author') == "John Doe"
        finally:
            os.unlink(temp_path)


class TestHTMLIntegration:
    """Integration tests for HTML extractor."""
    
    def test_simple_html_extraction(self):
        """Test extraction of simple HTML file."""
        from runeextract import extract
        
        content = """<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
</head>
<body>
    <h1>Main Heading</h1>
    <p>This is a paragraph.</p>
    <p>This is another paragraph.</p>
</body>
</html>
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name
        
        try:
            doc = extract(temp_path)
            
            assert doc.text
            assert doc.source_type == "html"
            assert "Main Heading" in doc.text
            assert doc.metadata.get('title') == "Test Page"
        finally:
            os.unlink(temp_path)
    
    def test_html_with_table(self):
        """Test extraction of HTML with table."""
        from runeextract import extract
        
        content = """<!DOCTYPE html>
<html>
<body>
    <table>
        <tr>
            <th>Name</th>
            <th>Age</th>
        </tr>
        <tr>
            <td>John</td>
            <td>25</td>
        </tr>
        <tr>
            <td>Jane</td>
            <td>30</td>
        </tr>
    </table>
</body>
</html>
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name
        
        try:
            doc = extract(temp_path)
            
            assert len(doc.tables) > 0
            table = doc.tables[0]
            assert "Name" in table.columns
            assert len(table.rows) == 2
        finally:
            os.unlink(temp_path)
    
    def test_html_with_meta_tags(self):
        """Test extraction of HTML with meta tags."""
        from runeextract import extract
        
        content = """<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
    <meta name="description" content="Test description">
    <meta name="keywords" content="test, html, extraction">
    <meta name="author" content="John Doe">
</head>
<body>
    <p>Content</p>
</body>
</html>
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name
        
        try:
            doc = extract(temp_path)
            
            assert doc.metadata.get('title') == "Test Page"
            assert doc.metadata.get('description') == "Test description"
            assert doc.metadata.get('keywords') == "test, html, extraction"
        finally:
            os.unlink(temp_path)


class TestChunkingIntegration:
    """Integration tests for chunking functionality."""
    
    def test_fixed_size_chunking(self):
        """Test fixed-size chunking with real content."""
        from runeextract import extract, ChunkingStrategy
        
        content = "A" * 2000
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name
        
        try:
            doc = extract(temp_path)
            chunks = doc.chunks(strategy=ChunkingStrategy.FIXED_SIZE, size=500, overlap=50)
            
            assert len(chunks) > 1
            assert all(chunk.text for chunk in chunks)
            assert all(chunk.chunk_id for chunk in chunks)
        finally:
            os.unlink(temp_path)
    
    def test_semantic_chunking(self):
        """Test semantic chunking with paragraphs."""
        from runeextract import extract, ChunkingStrategy
        
        content = """Paragraph 1 with some text.

Paragraph 2 with more text.

Paragraph 3 with additional text.
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name
        
        try:
            doc = extract(temp_path)
            chunks = doc.chunks(strategy=ChunkingStrategy.SEMANTIC, size=100)
            
            assert len(chunks) >= 1
            assert all(chunk.text for chunk in chunks)
        finally:
            os.unlink(temp_path)
    
    def test_chunk_caching(self):
        """Test that chunks are cached after first call."""
        from runeextract import extract, ChunkingStrategy
        
        content = "Test content for chunking."
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name
        
        try:
            doc = extract(temp_path)
            chunks1 = doc.chunks(strategy=ChunkingStrategy.FIXED_SIZE, size=100)
            chunks2 = doc.chunks(strategy=ChunkingStrategy.FIXED_SIZE, size=100)
            
            # Should return same cached chunks
            assert chunks1 is chunks2
        finally:
            os.unlink(temp_path)


class TestDocumentSerialization:
    """Integration tests for document serialization."""
    
    def test_document_to_dict(self):
        """Test document serialization to dictionary."""
        from runeextract import extract
        from runeextract.models.document import Table
        
        content = "# Test\n\nContent here."
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name
        
        try:
            doc = extract(temp_path)
            doc_dict = doc.to_dict()
            
            assert 'text' in doc_dict
            assert 'tables' in doc_dict
            assert 'images' in doc_dict
            assert 'metadata' in doc_dict
            assert 'source_type' in doc_dict
            assert doc_dict['source_type'] == "markdown"
        finally:
            os.unlink(temp_path)
    
    def test_document_with_table_to_dict(self):
        """Test document with table serialization."""
        from runeextract import extract
        
        content = """| A | B |
|---|---|
| 1 | 2 |
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name
        
        try:
            doc = extract(temp_path)
            doc_dict = doc.to_dict()
            
            assert len(doc_dict['tables']) > 0
            assert 'columns' in doc_dict['tables'][0]
            assert 'rows' in doc_dict['tables'][0]
        finally:
            os.unlink(temp_path)


class TestExtractManyIntegration:
    """Integration tests for batch extraction."""
    
    def test_extract_multiple_files(self):
        """Test extraction of multiple files."""
        from runeextract import extract_many
        
        # Create temporary files
        files = []
        for i in range(3):
            content = f"# Document {i}\n\nContent for document {i}."
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
                f.write(content)
                files.append(f.name)
        
        try:
            docs = extract_many(files)
            
            assert len(docs) == 3
            assert all(doc.text for doc in docs)
            assert all(doc.source_type == "markdown" for doc in docs)
        finally:
            for file_path in files:
                os.unlink(file_path)
    
    def test_extract_many_with_mixed_types(self):
        """Test extraction of files with different types."""
        from runeextract import extract_many
        
        files = []
        
        # Create markdown file
        md_content = "# Markdown"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(md_content)
            files.append(f.name)
        
        # Create HTML file
        html_content = "<html><body><p>HTML</p></body></html>"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(html_content)
            files.append(f.name)
        
        try:
            docs = extract_many(files)
            
            assert len(docs) == 2
            assert docs[0].source_type == "markdown"
            assert docs[1].source_type == "html"
        finally:
            for file_path in files:
                os.unlink(file_path)


class TestErrorHandlingIntegration:
    """Integration tests for error handling."""
    
    def test_nonexistent_file(self):
        """Test handling of non-existent file."""
        from runeextract import extract
        import pytest
        
        with pytest.raises(FileNotFoundError):
            extract("nonexistent_file.pdf")
    
    def test_unsupported_file_type(self):
        """Test handling of unsupported file type."""
        from runeextract import extract
        import pytest
        
        content = "test content"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name
        
        try:
            from runeextract.exceptions import UnsupportedFormatError
            with pytest.raises(UnsupportedFormatError):
                extract(temp_path)
        finally:
            os.unlink(temp_path)
