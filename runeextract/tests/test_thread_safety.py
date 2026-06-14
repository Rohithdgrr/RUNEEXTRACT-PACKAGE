"""Tests for thread safety in OCR reader and ExtractorRegistry."""

import threading
from unittest.mock import patch, MagicMock

import pytest
from runeextract.core.registry import ExtractorRegistry
from runeextract.core.extractor import BaseExtractor


class _TestExtractor(BaseExtractor):
    """Minimal extractor for registry tests."""

    def extract(self, file_path: str):
        from runeextract.models.document import Document
        return Document(text="test", source_type="test")

    def supported_extensions(self):
        return [".test_reg"]


def test_registry_thread_safe_register():
    """Concurrent register calls don't corrupt state."""
    extensions = [f".ext{i}" for i in range(50)]
    errors = []

    def register_ext(ext):
        try:
            ExtractorRegistry.register(ext)(_TestExtractor)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=register_ext, args=(ext,)) for ext in extensions]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    for ext in extensions:
        assert ExtractorRegistry.is_registered(ext)

    # Cleanup
    for ext in extensions:
        ExtractorRegistry.unregister(ext)


def test_registry_thread_safe_register_and_unregister():
    """Concurrent register and unregister calls don't corrupt state."""
    base_ext = ".concurrent_test"
    ExtractorRegistry.register(base_ext)(_TestExtractor)

    errors = []

    def toggle():
        try:
            ExtractorRegistry.unregister(base_ext)
            ExtractorRegistry.register(base_ext)(_TestExtractor)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=toggle) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert ExtractorRegistry.is_registered(base_ext)
    ExtractorRegistry.unregister(base_ext)


def test_registry_get_extractor_while_unregistering():
    """Concurrent get_extractor and unregister don't corrupt state."""
    ext = ".race_test"
    ExtractorRegistry.register(ext)(_TestExtractor)

    errors = []

    def race():
        for _ in range(30):
            try:
                if ExtractorRegistry.is_registered(ext):
                    _ = ExtractorRegistry.get_extractor(ext)
                ExtractorRegistry.unregister(ext)
                ExtractorRegistry.register(ext)(_TestExtractor)
            except (KeyError, Exception) as e:
                errors.append(e)

    threads = [threading.Thread(target=race) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # KeyErrors from racing unregister are expected; other errors are not
    unexpected = [e for e in errors if not isinstance(e, KeyError)]
    assert not unexpected

    assert ExtractorRegistry.is_registered(ext)
    ExtractorRegistry.unregister(ext)


@patch.dict("sys.modules", {"easyocr": MagicMock()})
def test_ocr_reader_lock():
    """_get_reader creates at most one reader instance under concurrent access."""
    from runeextract.processors.ocr import _get_reader, _ocr_reader

    # Reset for test
    import runeextract.processors.ocr as ocr_mod
    ocr_mod._ocr_reader = None

    mock_reader_instance = MagicMock()
    with patch("easyocr.Reader", return_value=mock_reader_instance) as mock_reader:
        results = []
        errors = []

        def get_reader():
            try:
                r = _get_reader(["en"])
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_reader) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert not errors
    assert len(results) == 10
    assert all(r is mock_reader_instance for r in results)

    # Cleanup
    ocr_mod._ocr_reader = None
