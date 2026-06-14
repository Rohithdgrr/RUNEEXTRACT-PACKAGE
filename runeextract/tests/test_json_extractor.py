"""Tests for JSON extractor."""

import os
import tempfile
import pytest
from runeextract import extract
from runeextract.exceptions import CorruptFileError


def test_json_dict():
    data = '{"name": "Alice", "age": 30, "city": "NYC"}'
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        f.write(data)
        path = f.name
    try:
        doc = extract(path)
        assert doc.source_type == "json"
        assert "Alice" in doc.text
        assert doc.metadata["type"] == "dict"
        assert "name" in doc.metadata.get("keys", [])
    finally:
        os.unlink(path)


def test_json_list_of_dicts():
    data = '[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]'
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        f.write(data)
        path = f.name
    try:
        doc = extract(path)
        assert doc.source_type == "json"
        assert doc.metadata["type"] == "list"
        assert doc.metadata["length"] == 2
        assert len(doc.tables) >= 1
    finally:
        os.unlink(path)


def test_json_empty_object():
    data = '{}'
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        f.write(data)
        path = f.name
    try:
        doc = extract(path)
        assert doc.text == "{}"
    finally:
        os.unlink(path)


def test_json_corrupt():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        f.write("{invalid json}")
        path = f.name
    try:
        with pytest.raises(CorruptFileError):
            extract(path)
    finally:
        os.unlink(path)
