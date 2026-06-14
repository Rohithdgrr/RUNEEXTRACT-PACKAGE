"""
Tests for document extractors.
"""

import pytest
from pathlib import Path


class TestPDFExtractor:
    """Tests for PDF extractor."""

    def test_supported_extensions(self):
        from runeextract.extractors.pdf.extractor import PDFExtractor
        extractor = PDFExtractor()
        assert ".pdf" in extractor.supported_extensions()

    def test_validate_nonexistent_file(self):
        from runeextract.extractors.pdf.extractor import PDFExtractor
        extractor = PDFExtractor()
        with pytest.raises(FileNotFoundError):
            extractor.validate_file("nonexistent.pdf")

    def test_validate_unsupported_extension(self):
        from runeextract.extractors.pdf.extractor import PDFExtractor
        extractor = PDFExtractor()
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            temp_path = f.name
        try:
            with pytest.raises(ValueError):
                extractor.validate_file(temp_path)
        finally:
            import os
            os.unlink(temp_path)


class TestDocxExtractor:
    """Tests for DOCX extractor."""

    def test_supported_extensions(self):
        from runeextract.extractors.docx.extractor import DocxExtractor
        extractor = DocxExtractor()
        assert ".docx" in extractor.supported_extensions()
        assert ".doc" in extractor.supported_extensions()


class TestPptxExtractor:
    """Tests for PPTX extractor."""

    def test_supported_extensions(self):
        from runeextract.extractors.pptx.extractor import PptxExtractor
        extractor = PptxExtractor()
        assert ".pptx" in extractor.supported_extensions()
        assert ".ppt" in extractor.supported_extensions()


class TestXlsxExtractor:
    """Tests for XLSX extractor."""

    def test_supported_extensions(self):
        from runeextract.extractors.xlsx.extractor import XlsxExtractor
        extractor = XlsxExtractor()
        assert ".xlsx" in extractor.supported_extensions()
        assert ".xls" in extractor.supported_extensions()


class TestHtmlExtractor:
    """Tests for HTML extractor."""

    def test_supported_extensions(self):
        from runeextract.extractors.html.extractor import HtmlExtractor
        extractor = HtmlExtractor()
        assert ".html" in extractor.supported_extensions()
        assert ".htm" in extractor.supported_extensions()


class TestMarkdownExtractor:
    """Tests for Markdown extractor."""

    def test_supported_extensions(self):
        from runeextract.extractors.markdown.extractor import MarkdownExtractor
        extractor = MarkdownExtractor()
        assert ".md" in extractor.supported_extensions()
        assert ".markdown" in extractor.supported_extensions()
