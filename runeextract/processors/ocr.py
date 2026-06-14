"""
OCR processor for images and scanned documents.

Uses easyocr (optional) with Pillow for image handling.
"""

import logging
import threading
from typing import Optional, List, Dict, Any
from io import BytesIO
from runeextract.exceptions import DependencyMissingError

logger = logging.getLogger(__name__)


_ocr_reader = None
_ocr_lock = threading.Lock()


def _get_reader(languages: List[str], gpu: bool = False) -> Any:
    global _ocr_reader
    if _ocr_reader is None:
        with _ocr_lock:
            if _ocr_reader is None:
                try:
                    import easyocr
                except ImportError:
                    raise DependencyMissingError("(image)", "easyocr")
                _ocr_reader = easyocr.Reader(languages, gpu=gpu)
    return _ocr_reader


def extract_text(
    image_data: bytes,
    languages: Optional[List[str]] = None,
    gpu: bool = False,
) -> str:
    """Run OCR on image bytes and return extracted text."""
    try:
        from PIL import Image as PILImage
    except ImportError:
        raise DependencyMissingError("(image)", "Pillow")

    langs = languages or ["en"]
    reader = _get_reader(langs, gpu=gpu)

    try:
        pil_image = PILImage.open(BytesIO(image_data))
        results = reader.readtext(pil_image)
        texts = [item[1] for item in results]
        return " ".join(texts)
    except Exception as exc:
        logger.warning(f"OCR failed: {exc}")
        return ""


def extract_text_with_boxes(
    image_data: bytes,
    languages: Optional[List[str]] = None,
    gpu: bool = False,
) -> List[Dict[str, Any]]:
    """Run OCR and return text with bounding boxes."""
    try:
        from PIL import Image as PILImage
    except ImportError:
        raise DependencyMissingError("(image)", "Pillow")

    langs = languages or ["en"]
    reader = _get_reader(langs, gpu=gpu)

    try:
        pil_image = PILImage.open(BytesIO(image_data))
        results = reader.readtext(pil_image)
        output = []
        for bbox, text, confidence in results:
            output.append({
                "text": text,
                "confidence": round(confidence, 3),
                "bbox": [[float(x), float(y)] for x, y in bbox],
            })
        return output
    except Exception as exc:
        logger.warning(f"OCR (detailed) failed: {exc}")
        return []
