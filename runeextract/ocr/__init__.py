"""Multi-language OCR detection and support."""

import re
from typing import Dict, List, Optional, Tuple


# Unicode range signatures for language family detection
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
}


LANG_TO_TESSERACT: Dict[str, str] = {
    "eng": "eng",
    "cmn": "chi_sim",
    "jpn": "jpn",
    "kor": "kor",
    "rus": "rus",
    "ara": "ara",
    "heb": "heb",
    "tha": "tha",
    "hin": "hin",
    "ell": "ell",
    "fra": "fra",
    "deu": "deu",
    "spa": "spa",
    "ita": "ita",
    "por": "por",
    "nld": "nld",
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

    def tesseract_langs(self, text: str) -> List[str]:
        lang = self.dominant(text)
        result = LANG_TO_TESSERACT.get(lang, "eng")
        return [result]

    def supported_languages(self) -> List[str]:
        return list(LANG_SIGNATURES.keys())


def detect_ocr_language(text: str) -> str:
    detector = OCRLanguageDetector()
    return detector.dominant(text)


def get_tesseract_langs(text: str) -> List[str]:
    detector = OCRLanguageDetector()
    return detector.tesseract_langs(text)


def get_ocr_languages() -> List[str]:
    return list(LANG_TO_TESSERACT.keys())
