"""
Image extractor for standalone image files.

Uses Pillow for metadata + easyocr (optional) for OCR.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from runeextract.core.extractor import BaseExtractor
from runeextract.models.document import Document as RuneDocument, Image as RuneImage
from runeextract.exceptions import CorruptFileError, DependencyMissingError

logger = logging.getLogger(__name__)


class ImageExtractor(BaseExtractor):
    """Extractor for standalone image files (.png, .jpg, .jpeg, .tiff, .bmp, .webp)."""

    def extract(self, file_path: str) -> RuneDocument:
        self.validate_file(file_path)

        text = ""
        images: List[RuneImage] = []
        metadata: Dict[str, Any] = {}

        try:
            from PIL import Image as PILImage
        except ImportError:
            raise DependencyMissingError(file_path, "Pillow")

        try:
            pil_img = PILImage.open(file_path)
            pil_img.load()
        except Exception as exc:
            raise CorruptFileError(file_path, detail=str(exc))

        img_format = pil_img.format or "unknown"
        width, height = pil_img.size
        mode = pil_img.mode

        metadata["width"] = width
        metadata["height"] = height
        metadata["format"] = img_format
        metadata["mode"] = mode

        with open(file_path, "rb") as f:
            image_data = f.read()

        images.append(RuneImage(
            data=image_data,
            format=img_format.lower(),
            width=width,
            height=height,
            metadata={"source": file_path}
        ))

        # OCR if enabled
        if self.ocr:
            try:
                from runeextract.processors.ocr import extract_text
                ocr_gpu = self.options.get("ocr_gpu", False)
                ocr_text = extract_text(image_data, gpu=ocr_gpu)
                if ocr_text:
                    text = ocr_text
                    metadata["ocr"] = True
                    metadata["ocr_languages"] = self.options.get("ocr_languages", ["en"])
                    if ocr_gpu:
                        metadata["ocr_gpu"] = True
            except DependencyMissingError:
                logger.warning("OCR requested but easyocr not installed")
            except Exception as exc:
                logger.warning(f"OCR failed for {file_path}: {exc}")

        text = self.clean_text(text)

        return RuneDocument(
            text=text, tables=[], images=images, metadata=metadata,
            source_type="image", source_path=file_path
        )

    def supported_extensions(self) -> list[str]:
        return [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"]
