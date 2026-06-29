"""Multi-language OCR detection and support."""

import re
from typing import Dict, List, Optional, Tuple

LANG_SIGNATURES: Dict[str, List[Tuple[str, str]]] = {
    "eng": [("Latin", r"^[A-Za-z0-9\s\.\,\;\:\!\?\"\'\-\_\(\)\[\]\{\}/\\@#$%^&*+=<>]+$")],
    "cmn": [("CJK", r"[\u4e00-\u9fff\u3400-\u4dbf]")],
    "jpn": [("CJK", r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]")],
    "kor": [("Hangul", r"[\uac00-\ud7af\u1100-\u11ff]")],
    "rus": [("Cyrillic", r"[\u0400-\u04ff]")],
    "ara": [("Arabic", r"[\u0600-\u06ff\u0750-\u077f]")],
    "heb": [("Hebrew", r"[\u0590-\u05ff]")],
    "tha": [("Thai", r"[\u0e00-\u0e7f]")],
    "hin": [("Devanagari", r"[\u0900-\u097f]")],
    "ell": [("Greek", r"[\u0370-\u03ff]")],
    "fra": [("LatinExtA", r"[\u00c0-\u00ff]")],
    "deu": [("LatinExtA", r"[\u00c0-\u00ff]")],
    "spa": [("LatinExtA", r"[\u00c0-\u00ff]")],
    "ita": [("LatinExtA", r"[\u00c0-\u00ff]")],
    "por": [("LatinExtA", r"[\u00c0-\u00ff]")],
    "vie": [("LatinExtB", r"[\u01a0-\u01b0\u1ea0-\u1ef9]")],
    "tur": [("LatinExtA", r"[\u0150-\u015f\u00d6\u00dc\u0130\u0131]")],
    "pol": [("LatinExtA", r"[\u0104-\u0179]")],
}

LANG_TO_TESSERACT: Dict[str, str] = {
    "eng": "eng", "cmn": "chi_sim", "jpn": "jpn", "kor": "kor",
    "rus": "rus", "ara": "ara", "heb": "heb", "tha": "tha",
    "hin": "hin", "ell": "ell", "fra": "fra", "deu": "deu",
    "spa": "spa", "ita": "ita", "por": "por", "nld": "nld",
    "vie": "vie", "tur": "tur", "pol": "pol",
}


class OCRLanguageDetector:
    def detect(self, text: str) -> List[Tuple[str, float]]:
        if not text or not text.strip():
            return [("eng", 1.0)]
        scores: Dict[str, float] = {}
        total = max(len(text), 1)
        for lang, signatures in LANG_SIGNATURES.items():
            score = 0.0
            for _, pattern in signatures:
                matches = len(re.findall(pattern, text))
                score += matches / total
            if score > 0:
                scores[lang] = score
        if not scores:
            return [("eng", 1.0)]
        total_score = sum(scores.values())
        ranked = sorted(
            [(lang, s / total_score) for lang, s in scores.items()],
            key=lambda x: -x[1],
        )
        return ranked

    def dominant(self, text: str) -> str:
        ranked = self.detect(text)
        return ranked[0][0] if ranked else "eng"

    def multi_lang(self, text: str, threshold: float = 0.1) -> List[str]:
        return [lang for lang, score in self.detect(text) if score >= threshold]

    def tesseract_langs(self, text: str) -> List[str]:
        langs = self.multi_lang(text, threshold=0.05)
        results = set()
        for lang in langs:
            result = LANG_TO_TESSERACT.get(lang)
            if result:
                results.add(result)
        if not results:
            lang = self.dominant(text)
            return [LANG_TO_TESSERACT.get(lang, "eng")]
        return sorted(results)

    def supported_languages(self) -> List[str]:
        return list(LANG_SIGNATURES.keys())


def detect_ocr_language(text: str) -> str:
    return OCRLanguageDetector().dominant(text)


def get_tesseract_langs(text: str) -> List[str]:
    return OCRLanguageDetector().tesseract_langs(text)


def get_ocr_languages() -> List[str]:
    return list(LANG_TO_TESSERACT.keys())


def detect_text_script(text: str) -> str:
    """Detect the dominant writing script (Latin, CJK, Cyrillic, etc.)."""
    detector = OCRLanguageDetector()
    lang = detector.dominant(text)
    for scripts in LANG_SIGNATURES.values():
        for script_name, _ in scripts:
            if any(lang == k for k in LANG_SIGNATURES if k == lang):
                return script_name
    return "Latin"
