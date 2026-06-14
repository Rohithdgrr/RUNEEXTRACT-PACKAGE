"""
Tests for extractor router.
"""

import pytest
from runeextract.core.router import ExtractorRouter


def test_supported_extensions():
    """Test getting supported extensions."""
    extensions = ExtractorRouter.supported_extensions()
    
    assert ".pdf" in extensions
    assert ".docx" in extensions
    assert ".html" in extensions
    assert ".md" in extensions
    assert ".pptx" in extensions
    assert ".xlsx" in extensions
    assert ".csv" in extensions
    assert ".json" in extensions
    assert ".png" in extensions
    assert ".epub" in extensions


def test_get_source_type():
    """Test getting source type from file path."""
    assert ExtractorRouter.get_source_type("test.pdf") == "pdf"
    assert ExtractorRouter.get_source_type("test.docx") == "docx"
    assert ExtractorRouter.get_source_type("test.html") == "html"
    assert ExtractorRouter.get_source_type("test.md") == "markdown"
    assert ExtractorRouter.get_source_type("test.markdown") == "markdown"
    assert ExtractorRouter.get_source_type("test.pptx") == "pptx"
    assert ExtractorRouter.get_source_type("test.xlsx") == "xlsx"
    assert ExtractorRouter.get_source_type("test.csv") == "csv"
    assert ExtractorRouter.get_source_type("test.json") == "json"
    assert ExtractorRouter.get_source_type("test.png") == "image"
    assert ExtractorRouter.get_source_type("test.epub") == "epub"


def test_get_extractor_unsupported():
    """Test getting extractor for unsupported file type."""
    from runeextract.exceptions import UnsupportedFormatError
    with pytest.raises(UnsupportedFormatError):
        ExtractorRouter.get_extractor("test.xyz")
