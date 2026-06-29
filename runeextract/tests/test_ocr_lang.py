"""Tests for multi-language OCR detection."""

import pytest

from runeextract.ocr import (
    OCRLanguageDetector, detect_ocr_language,
    get_tesseract_langs, get_ocr_languages,
)


class TestOCRLanguageDetector:
    def test_detect_english(self):
        d = OCRLanguageDetector()
        result = d.detect("Hello world, this is English text.")
        assert result[0][0] == "eng"

    def test_detect_chinese(self):
        d = OCRLanguageDetector()
        result = d.detect("你好世界，这是中文文本。")
        top = max(result, key=lambda x: x[1])
        assert top[0] == "cmn"

    def test_detect_japanese(self):
        d = OCRLanguageDetector()
        result = d.detect("こんにちは世界 これは日本語です")
        top = max(result, key=lambda x: x[1])
        assert top[0] == "jpn"

    def test_detect_korean(self):
        d = OCRLanguageDetector()
        result = d.detect("안녕하세요 세상 이것은 한국어입니다")
        top = max(result, key=lambda x: x[1])
        assert top[0] == "kor"

    def test_detect_russian(self):
        d = OCRLanguageDetector()
        result = d.detect("Привет мир, это русский текст")
        top = max(result, key=lambda x: x[1])
        assert top[0] == "rus"

    def test_detect_arabic(self):
        d = OCRLanguageDetector()
        result = d.detect("مرحبا بالعالم هذا نص عربي")
        top = max(result, key=lambda x: x[1])
        assert top[0] == "ara"

    def test_detect_hebrew(self):
        d = OCRLanguageDetector()
        result = d.detect("שלום עולם זה טקסט בעברית")
        top = max(result, key=lambda x: x[1])
        assert top[0] == "heb"

    def test_empty_text(self):
        d = OCRLanguageDetector()
        assert d.detect("")[0][0] == "eng"
        assert d.detect("   ")[0][0] == "eng"

    def test_dominant_english(self):
        d = OCRLanguageDetector()
        assert d.dominant("Hello world") == "eng"

    def test_dominant_chinese(self):
        d = OCRLanguageDetector()
        lang = d.dominant("你好世界")
        assert lang == "cmn"

    def test_tesseract_langs_english(self):
        d = OCRLanguageDetector()
        assert d.tesseract_langs("Hello") == ["eng"]

    def test_tesseract_langs_chinese(self):
        d = OCRLanguageDetector()
        langs = d.tesseract_langs("你好")
        assert "chi_sim" in langs

    def test_supported_languages(self):
        d = OCRLanguageDetector()
        langs = d.supported_languages()
        assert "eng" in langs
        assert "cmn" in langs
        assert "jpn" in langs


class TestDetectOCRLanguage:
    def test_convenience(self):
        assert detect_ocr_language("Hello") == "eng"
        assert detect_ocr_language("你好") == "cmn"


class TestGetTesseractLangs:
    def test_convenience(self):
        assert get_tesseract_langs("Hello") == ["eng"]


class TestGetOCRLanguages:
    def test_all(self):
        langs = get_ocr_languages()
        assert "eng" in langs
        assert "fra" in langs
        assert "deu" in langs
