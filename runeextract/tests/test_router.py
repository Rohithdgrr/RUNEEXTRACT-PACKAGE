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


def test_get_source_type():
    """Test getting source type from file path."""
    assert ExtractorRouter.get_source_type("test.pdf") == "pdf"
    assert ExtractorRouter.get_source_type("test.docx") == "docx"
    assert ExtractorRouter.get_source_type("test.html") == "html"
    assert ExtractorRouter.get_source_type("test.md") == "markdown"


def test_get_extractor_unsupported():
    """Test getting extractor for unsupported file type."""
    with pytest.raises(ValueError):
        ExtractorRouter.get_extractor("test.xyz")
