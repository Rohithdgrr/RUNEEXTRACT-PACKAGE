"""
Tests for document extractors.
"""

import pytest
from pathlib import Path
from runeextract.extractors.pdf.extractor import PDFExtractor
from runeextract.extractors.docx.extractor import DocxExtractor
from runeextract.extractors.html.extractor import HtmlExtractor
from runeextract.extractors.markdown.extractor import MarkdownExtractor


class TestPDFExtractor:
    """Tests for PDF extractor."""
    
    def test_supported_extensions(self):
        """Test supported file extensions."""
        extractor = PDFExtractor()
        assert ".pdf" in extractor.supported_extensions()
    
    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        extractor = PDFExtractor()
        with pytest.raises(FileNotFoundError):
            extractor.validate_file("nonexistent.pdf")
    
    def test_validate_unsupported_extension(self):
        """Test validation of unsupported file extension."""
        extractor = PDFExtractor()
        with pytest.raises(ValueError):
            extractor.validate_file("test.txt")


class TestDocxExtractor:
    """Tests for DOCX extractor."""
    
    def test_supported_extensions(self):
        """Test supported file extensions."""
        extractor = DocxExtractor()
        assert ".docx" in extractor.supported_extensions()
        assert ".doc" in extractor.supported_extensions()


class TestHtmlExtractor:
    """Tests for HTML extractor."""
    
    def test_supported_extensions(self):
        """Test supported file extensions."""
        extractor = HtmlExtractor()
        assert ".html" in extractor.supported_extensions()
        assert ".htm" in extractor.supported_extensions()


class TestMarkdownExtractor:
    """Tests for Markdown extractor."""
    
    def test_supported_extensions(self):
        """Test supported file extensions."""
        extractor = MarkdownExtractor()
        assert ".md" in extractor.supported_extensions()
        assert ".markdown" in extractor.supported_extensions()
