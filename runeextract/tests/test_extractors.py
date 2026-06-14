"""
Tests for document extractors.
"""

import os
import tempfile
from unittest.mock import patch

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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("non-empty content")
            temp_path = f.name
        try:
            from runeextract.exceptions import UnsupportedFormatError
            with pytest.raises(UnsupportedFormatError):
                extractor.validate_file(temp_path)
        finally:
            os.unlink(temp_path)

    def test_validate_empty_file(self):
        from runeextract.extractors.pdf.extractor import PDFExtractor
        extractor = PDFExtractor()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            temp_path = f.name
        try:
            from runeextract.exceptions import CorruptFileError
            with pytest.raises(CorruptFileError):
                extractor.validate_file(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_validate_directory(self):
        from runeextract.extractors.pdf.extractor import PDFExtractor
        from runeextract.exceptions import ExtractionError
        extractor = PDFExtractor()
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ExtractionError) as excinfo:
                extractor.validate_file(tmpdir)
            assert excinfo.value.error_code == "E041"

    def test_validate_no_permission(self):
        from runeextract.extractors.pdf.extractor import PDFExtractor
        from runeextract.exceptions import ExtractionError
        extractor = PDFExtractor()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdf", delete=False) as f:
            f.write("content")
            temp_path = f.name
        try:
            with patch("os.access", return_value=False):
                with pytest.raises(ExtractionError) as excinfo:
                    extractor.validate_file(temp_path)
                assert excinfo.value.error_code == "E040"
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_validate_stat_fails(self):
        from runeextract.extractors.pdf.extractor import PDFExtractor
        from runeextract.exceptions import ExtractionError
        extractor = PDFExtractor()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdf", delete=False) as f:
            f.write("content")
            temp_path = f.name
        try:
            original_stat = Path.stat
            def broken_stat(self_):
                raise OSError("stat failed")
            with patch.object(Path, "stat", broken_stat):
                with pytest.raises(ExtractionError) as excinfo:
                    extractor.validate_file(temp_path)
                assert excinfo.value.error_code == "E042"
        finally:
            if os.path.exists(temp_path):
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


class TestCsvExtractor:
    """Tests for CSV extractor."""

    def test_supported_extensions(self):
        from runeextract.extractors.csv.extractor import CsvExtractor
        extractor = CsvExtractor()
        assert ".csv" in extractor.supported_extensions()


class TestJsonExtractor:
    """Tests for JSON extractor."""

    def test_supported_extensions(self):
        from runeextract.extractors.json.extractor import JsonExtractor
        extractor = JsonExtractor()
        assert ".json" in extractor.supported_extensions()


class TestImageExtractor:
    """Tests for Image extractor."""

    def test_supported_extensions(self):
        from runeextract.extractors.image.extractor import ImageExtractor
        extractor = ImageExtractor()
        assert ".png" in extractor.supported_extensions()
        assert ".jpg" in extractor.supported_extensions()
        assert ".jpeg" in extractor.supported_extensions()


class TestEpubExtractor:
    """Tests for EPUB extractor."""

    def test_supported_extensions(self):
        from runeextract.extractors.epub.extractor import EpubExtractor
        extractor = EpubExtractor()
        assert ".epub" in extractor.supported_extensions()
