"""Tests for streaming extraction."""

import os
import tempfile
import pytest
from runeextract.core.streaming import get_streaming_extractor
from runeextract.core.extractor import StreamingExtractor
from runeextract.extractors.pdf.extractor import PdfStreamingExtractor
from runeextract.exceptions import UnsupportedFormatError


def test_pdf_streaming_extractor_class():
    assert issubclass(PdfStreamingExtractor, StreamingExtractor)


def test_streaming_extractor_has_extract_stream():
    import inspect
    assert inspect.isasyncgenfunction(StreamingExtractor.extract_stream)


def test_streaming_fallback_markdown():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Hello")
        path = f.name
    try:
        ext = get_streaming_extractor(path)
        assert ext is not None
        assert hasattr(ext, "extract_stream")
    finally:
        os.unlink(path)


def test_streaming_pdf_extractor():
    ext = get_streaming_extractor("dummy.pdf", ocr=False)
    assert ext is not None


def test_streaming_unsupported_fallback():
    with pytest.raises(UnsupportedFormatError):
        get_streaming_extractor("dummy.xyz")


def test_streaming_wrapped_supported_extensions():
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("a,b\n1,2")
        path = f.name
    try:
        ext = get_streaming_extractor(path)
        assert ".csv" in ext.supported_extensions()
    finally:
        os.unlink(path)


def test_streaming_wrapped_extract():
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Streaming Wrapped Test")
        path = f.name
    try:
        ext = get_streaming_extractor(path)
        doc = ext.extract(path)
        assert doc is not None
        assert "Streaming Wrapped Test" in doc.text
    finally:
        os.unlink(path)
