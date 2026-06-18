"""Tests for Fast Mode / Quality Levels."""

import pytest

from runeextract.quality import FastMode, QualityLevel, configure_quality


class TestQualityLevel:
    def test_values(self):
        assert QualityLevel.FAST.value == "fast"
        assert QualityLevel.MAXIMUM.value == "maximum"


class TestFastMode:
    def test_defaults(self):
        fm = FastMode()
        assert fm.enabled is False
        assert fm.skip_ocr is True
        assert fm.extract_timeout == 30

    def test_for_level_fast(self):
        fm = FastMode.for_level(QualityLevel.FAST)
        assert fm.enabled is True
        assert fm.skip_ocr is True
        assert fm.skip_tables is True
        assert fm.max_pages == 5
        assert fm.extract_timeout == 15

    def test_for_level_standard(self):
        fm = FastMode.for_level(QualityLevel.STANDARD)
        assert fm.enabled is False
        assert fm.skip_ocr is False
        assert fm.max_pages is None
        assert fm.extract_timeout == 60

    def test_for_level_high(self):
        fm = FastMode.for_level(QualityLevel.HIGH)
        assert fm.enabled is False
        assert fm.extract_timeout == 300

    def test_for_level_maximum(self):
        fm = FastMode.for_level(QualityLevel.MAXIMUM)
        assert fm.enabled is False
        assert fm.extract_timeout == 600

    def test_custom_override(self):
        fm = FastMode(enabled=True, extract_timeout=10, max_pages=2)
        assert fm.extract_timeout == 10
        assert fm.max_pages == 2

    def test_unknown_level_falls_back(self):
        fm = FastMode.for_level("unknown")  # type: ignore
        assert fm.enabled is False


class TestConfigureQuality:
    def test_fast(self):
        fm = configure_quality(QualityLevel.FAST)
        assert fm.enabled is True
        assert fm._level == QualityLevel.FAST

    def test_standard(self):
        fm = configure_quality(QualityLevel.STANDARD)
        assert fm.enabled is False
