"""Tests for extraction cache."""

import os
import tempfile
from runeextract.core.cache import ExtractionCache
from runeextract.models.document import Document


def test_cache_init():
    cache = ExtractionCache()
    assert cache is not None
    cache.close()


def test_cache_set_get():
    cache = ExtractionCache()
    path = __file__
    opts = {"ocr": False}
    doc = Document(text="hello", source_type="text", source_path=path)
    cache.set(path, opts, doc)
    result = cache.get(path, opts)
    assert result is not None
    assert result.text == "hello"
    cache.close()


def test_cache_invalidate():
    cache = ExtractionCache()
    path = __file__
    opts = {"ocr": False}
    doc = Document(text="data", source_type="text", source_path=path)
    cache.set(path, opts, doc)
    assert cache.get(path, opts) is not None
    cache.invalidate(path)
    assert cache.get(path, opts) is None
    cache.close()


def test_cache_key_hashing():
    cache = ExtractionCache()
    key1 = cache._make_key("file1.pdf", {})
    key2 = cache._make_key("file1.pdf", {})
    key3 = cache._make_key("file2.pdf", {})
    assert key1 == key2
    assert key1 != key3
    cache.close()
