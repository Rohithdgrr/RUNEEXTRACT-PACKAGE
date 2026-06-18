"""Tests for pre-signed URL extraction."""

import os
import tempfile
import pytest

from runeextract.storage.presigned import extract_from_presigned_url, _infer_filename


class TestInferFilename:
    def test_from_path(self):
        url = "https://s3.amazonaws.com/bucket/report.pdf?Signature=abc"
        name = _infer_filename(url, "application/pdf")
        assert name == "report.pdf"

    def test_from_content_type(self):
        url = "https://s3.amazonaws.com/bucket/doc?Signature=abc"
        name = _infer_filename(url, "text/csv")
        assert name.endswith(".csv") or name == "document.csv"

    def test_fallback(self):
        url = "https://example.com/file"
        name = _infer_filename(url, "application/octet-stream")
        assert name == "document.bin"

    def test_url_decoded_path(self):
        url = "https://example.com/my%20file.pdf"
        name = _infer_filename(url, "application/pdf")
        assert name == "my file.pdf"


class TestExtractFromPresignedUrl:
    def test_invalid_url(self):
        with pytest.raises(Exception):
            extract_from_presigned_url("ftp://invalid")

    def test_nonexistent_url(self):
        with pytest.raises(Exception):
            extract_from_presigned_url("https://nonexistent.example.com/file.pdf")

    def test_bad_url_format(self):
        with pytest.raises(Exception):
            extract_from_presigned_url("not-a-url-at-all")
