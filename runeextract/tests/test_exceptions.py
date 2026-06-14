"""
Tests for custom exceptions.
"""

import pytest
from runeextract.exceptions import (
    ExtractionError, UnsupportedFormatError, CorruptFileError,
    FileTooLargeError, DependencyMissingError
)


def test_extraction_error():
    err = ExtractionError("boom", file_path="f.pdf")
    assert "E000" in str(err)
    assert "f.pdf" in str(err)


def test_unsupported_format_error():
    err = UnsupportedFormatError("f.xyz", extension=".xyz")
    assert "E001" in str(err)
    assert "f.xyz" in str(err)


def test_corrupt_file_error():
    err = CorruptFileError("f.pdf", detail="bad header")
    assert "E002" in str(err)
    assert "bad header" in str(err)


def test_file_too_large_error():
    err = FileTooLargeError("f.pdf", size=600_000_000, limit=500_000_000)
    assert "E003" in str(err)
    assert "600,000,000" in str(err)


def test_dependency_missing_error():
    err = DependencyMissingError("f.pdf", "tesseract")
    assert "E004" in str(err)
    assert "tesseract" in str(err)
