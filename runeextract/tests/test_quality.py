"""Tests for Quality Levels / Fast Mode feature."""

import pytest

from runeextract.quality import QualityLevel, QualityConfig, get_quality_config, extract_with_quality, resolution_map


class TestQualityLevel:
    def test_values(self):
        assert QualityLevel.DRAFT.value == "draft"
        assert QualityLevel.NORMAL.value == "normal"
        assert QualityLevel.VERIFIED.value == "verified"

    def test_enum_members(self):
        assert list(QualityLevel) == [QualityLevel.DRAFT, QualityLevel.NORMAL, QualityLevel.VERIFIED]


class TestQualityConfig:
    def test_draft_config(self):
        config = get_quality_config(QualityLevel.DRAFT)
        assert config.ocr is False
        assert config.tables is False
        assert config.images is False

    def test_normal_config(self):
        config = get_quality_config("normal")
        assert config.ocr is True
        assert config.tables is True
        assert config.image_dpi == 150

    def test_verified_config(self):
        config = get_quality_config(QualityLevel.VERIFIED)
        assert config.ocr is True
        assert config.ai_analysis is True
        assert config.image_dpi == 300

    def test_get_quality_config_invalid(self):
        with pytest.raises(ValueError):
            get_quality_config("nonexistent")

    def test_to_extract_kwargs(self):
        config = QualityConfig(ocr=True, tables=False)
        kwargs = config.to_extract_kwargs()
        assert kwargs["ocr"] is True
        assert kwargs["tables"] is False
        assert kwargs["images"] is True

    def test_resolution_map(self):
        assert resolution_map[QualityLevel.DRAFT] == 72
        assert resolution_map[QualityLevel.NORMAL] == 150
        assert resolution_map[QualityLevel.VERIFIED] == 300


class TestExtractWithQuality:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            extract_with_quality("/nonexistent/file.pdf", level=QualityLevel.DRAFT)
