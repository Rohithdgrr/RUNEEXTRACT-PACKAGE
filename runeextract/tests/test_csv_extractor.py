"""Tests for CSV extractor."""

import os
import tempfile
import pytest
from runeextract import extract
from runeextract.exceptions import CorruptFileError, UnsupportedFormatError


def test_csv_basic_extraction():
    csv_content = "name,age,city\nAlice,30,NYC\nBob,25,LA\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(csv_content)
        path = f.name
    try:
        doc = extract(path)
        assert doc.source_type == "csv"
        assert "Alice" in doc.text
        assert "Bob" in doc.text
        assert doc.metadata["row_count"] == 3
        assert doc.metadata["column_count"] == 3
        assert len(doc.tables) == 1
        assert doc.tables[0].columns == ["name", "age", "city"]
        assert len(doc.tables[0].rows) == 2
    finally:
        os.unlink(path)


def test_csv_empty():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        path = f.name
    try:
        with pytest.raises(CorruptFileError):
            extract(path)
    finally:
        os.unlink(path)


def test_csv_single_row():
    csv_content = "header_only\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(csv_content)
        path = f.name
    try:
        doc = extract(path)
        assert "header_only" in doc.text
    finally:
        os.unlink(path)


def test_csv_corrupt():
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as f:
        f.write(b"\xff\xfe\x00\x01")
        path = f.name
    try:
        with pytest.raises(CorruptFileError):
            extract(path)
    finally:
        os.unlink(path)
