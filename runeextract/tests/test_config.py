"""Tests for configuration system."""

import os
import json
import tempfile
import pytest
from runeextract.config import RuneExtractConfig, get_config, set_config


def test_config_defaults():
    cfg = RuneExtractConfig()
    assert cfg.ocr is False
    assert cfg.tables is True
    assert cfg.images is True
    assert cfg.metadata is True
    assert cfg.max_file_size == 500 * 1024 * 1024
    assert cfg.extra == {}


def test_config_with_values():
    cfg = RuneExtractConfig(ocr=True, tables=False, max_file_size=1000)
    assert cfg.ocr is True
    assert cfg.tables is False
    assert cfg.max_file_size == 1000


def test_config_merge_options():
    cfg = RuneExtractConfig(ocr=False, tables=True)
    merged = cfg.merge_options(ocr=True, chunk_size=500)
    assert merged.ocr is True
    assert merged.tables is True
    assert merged.chunk_size == 500
    assert cfg.ocr is False


def test_config_to_dict():
    cfg = RuneExtractConfig(ocr=True)
    d = cfg.to_dict()
    assert d["ocr"] is True
    assert d["max_file_size"] == 500 * 1024 * 1024


def test_config_env_override(monkeypatch):
    cfg = RuneExtractConfig(ocr=False, tables=True)
    monkeypatch.setenv("RUNEEXTRACT_OCR", "true")
    monkeypatch.setenv("RUNEEXTRACT_MAX_FILE_SIZE", "999")
    cfg.apply_env()
    assert cfg.ocr is True
    assert cfg.max_file_size == 999


def test_config_env_override_bool_false(monkeypatch):
    cfg = RuneExtractConfig(tables=True, images=True)
    monkeypatch.setenv("RUNEEXTRACT_TABLES", "false")
    monkeypatch.setenv("RUNEEXTRACT_IMAGES", "0")
    cfg.apply_env()
    assert cfg.tables is False
    assert cfg.images is False


def test_config_merge_extra():
    cfg = RuneExtractConfig()
    merged = cfg.merge_options(ai_model="gpt-4", ai_temperature=0.5)
    assert merged.extra == {"ai_model": "gpt-4", "ai_temperature": 0.5}
    assert merged.ocr is False  # base field unchanged
    assert merged.chunk_size == 1000  # base field unchanged


def test_get_set_config():
    original = get_config()
    assert original is not None
    custom = RuneExtractConfig(ocr=True)
    set_config(custom)
    assert get_config() is custom
    set_config(original)
