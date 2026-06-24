"""Fast mode / quality level configuration for extraction speed vs quality tradeoffs."""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional


class QualityLevel(str, Enum):
    FAST = "fast"
    STANDARD = "standard"
    HIGH = "high"
    MAXIMUM = "maximum"


@dataclass
class FastMode:
    enabled: bool = False
    skip_ocr: bool = True
    skip_images: bool = True
    skip_tables: bool = False
    skip_metadata: bool = False
    skip_chunking: bool = True
    skip_embeddings: bool = True
    max_pages: Optional[int] = None
    extract_timeout: int = 30
    _level: QualityLevel = QualityLevel.STANDARD

    @classmethod
    def for_level(cls, level: QualityLevel) -> "FastMode":
        configs: Dict[QualityLevel, dict] = {
            QualityLevel.FAST: {
                "enabled": True,
                "skip_ocr": True,
                "skip_images": True,
                "skip_tables": True,
                "skip_metadata": True,
                "skip_chunking": True,
                "skip_embeddings": True,
                "max_pages": 5,
                "extract_timeout": 15,
                "_level": QualityLevel.FAST,
            },
            QualityLevel.STANDARD: {
                "enabled": False,
                "skip_ocr": False,
                "skip_images": False,
                "skip_tables": False,
                "skip_metadata": False,
                "skip_chunking": False,
                "skip_embeddings": False,
                "max_pages": None,
                "extract_timeout": 60,
                "_level": QualityLevel.STANDARD,
            },
            QualityLevel.HIGH: {
                "enabled": False,
                "skip_ocr": False,
                "skip_images": False,
                "skip_tables": False,
                "skip_metadata": False,
                "skip_chunking": False,
                "skip_embeddings": False,
                "extract_timeout": 300,
                "_level": QualityLevel.HIGH,
            },
            QualityLevel.MAXIMUM: {
                "enabled": False,
                "skip_ocr": False,
                "skip_images": False,
                "skip_tables": False,
                "skip_metadata": False,
                "skip_chunking": False,
                "skip_embeddings": False,
                "extract_timeout": 600,
                "_level": QualityLevel.MAXIMUM,
            },
        }
        return cls(**configs.get(level, configs[QualityLevel.STANDARD]))


def configure_quality(level: object = None, **kwargs) -> FastMode:
    if level is None and "level" in kwargs:
        level = kwargs.pop("level")
    if isinstance(level, str):
        level = QualityLevel(level)
    if level is None:
        level = QualityLevel.STANDARD
    return FastMode.for_level(level)
