"""Tests for password-protected file support (PDF, DOCX, XLSX)."""

import os
import tempfile
import pytest

from runeextract import extract
from runeextract.exceptions import ExtractionError, WrongPasswordError


def _create_encrypted_pdf(password: str = "secret123") -> str:
    """Create a small password-protected PDF using PyMuPDF."""
    import fitz
    path = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Protected PDF content")
    doc.save(path, encryption=4, user_pw=password, owner_pw=password)
    doc.close()
    return path


def _create_plain_pdf() -> str:
    """Create a small plain PDF."""
    import fitz
    path = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Plain PDF content")
    doc.save(path)
    doc.close()
    return path


class TestPasswordProtectedPDF:
    def test_extract_with_correct_password(self):
        path = _create_encrypted_pdf("opensesame")
        try:
            doc = extract(path, password="opensesame")
            assert "Protected PDF content" in doc.text
        finally:
            os.unlink(path)

    def test_extract_without_password_raises(self):
        path = _create_encrypted_pdf("secret123")
        try:
            with pytest.raises(WrongPasswordError):
                extract(path)
        finally:
            os.unlink(path)

    def test_extract_with_wrong_password_raises(self):
        path = _create_encrypted_pdf("realpass")
        try:
            with pytest.raises(WrongPasswordError):
                extract(path, password="wrong")
        finally:
            os.unlink(path)

    def test_plain_pdf_no_password_needed(self):
        path = _create_plain_pdf()
        try:
            doc = extract(path)
            assert "Plain PDF content" in doc.text
        finally:
            os.unlink(path)

    def test_plain_pdf_with_password_ignored(self):
        path = _create_plain_pdf()
        try:
            doc = extract(path, password="ignored")
            assert "Plain PDF content" in doc.text
        finally:
            os.unlink(path)

    def test_extract_tables_with_password(self):
        path = _create_encrypted_pdf("pw")
        try:
            doc = extract(path, password="pw")
            assert doc.tables is not None
        finally:
            os.unlink(path)


class TestPasswordProtectedDOCX:
    def test_open_protected_missing_dep(self):
        from runeextract.extractors.docx.extractor import DocxExtractor
        with pytest.raises(ExtractionError, match="msoffcrypto"):
            DocxExtractor._open_protected("/nonexistent.docx", "pw")


class TestPasswordProtectedXLSX:
    def test_open_protected_missing_dep(self):
        from runeextract.extractors.xlsx.extractor import XlsxExtractor
        with pytest.raises(ExtractionError, match="msoffcrypto"):
            XlsxExtractor._open_protected("/nonexistent.xlsx", "pw")
