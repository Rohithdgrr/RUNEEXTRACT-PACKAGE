"""Quality level presets for document extraction."""

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class QualityLevel(str, enum.Enum):
    DRAFT = "draft"
    NORMAL = "normal"
    VERIFIED = "verified"


resolution_map = {
    QualityLevel.DRAFT: 72,
    QualityLevel.NORMAL: 150,
    QualityLevel.VERIFIED: 300,
}


@dataclass
class QualityConfig:
    level: QualityLevel = QualityLevel.NORMAL
    ocr: bool = False
    tables: bool = True
    images: bool = True
    metadata: bool = True
    chunking: bool = False
    ai_analysis: bool = False
    image_dpi: int = 150

    def to_extract_kwargs(self) -> Dict[str, Any]:
        return {
            "ocr": self.ocr,
            "tables": self.tables,
            "images": self.images,
            "metadata": self.metadata,
        }


DRAFT_CONFIG = QualityConfig(
    level=QualityLevel.DRAFT,
    ocr=False,
    tables=False,
    images=False,
    metadata=False,
    chunking=False,
    ai_analysis=False,
    image_dpi=72,
)

NORMAL_CONFIG = QualityConfig(
    level=QualityLevel.NORMAL,
    ocr=True,
    tables=True,
    images=True,
    metadata=True,
    chunking=False,
    ai_analysis=False,
    image_dpi=150,
)

VERIFIED_CONFIG = QualityConfig(
    level=QualityLevel.VERIFIED,
    ocr=True,
    tables=True,
    images=True,
    metadata=True,
    chunking=True,
    ai_analysis=True,
    image_dpi=300,
)

_PRESETS = {
    QualityLevel.DRAFT: DRAFT_CONFIG,
    QualityLevel.NORMAL: NORMAL_CONFIG,
    QualityLevel.VERIFIED: VERIFIED_CONFIG,
}


def get_quality_config(level: QualityLevel) -> QualityConfig:
    if isinstance(level, str):
        level = QualityLevel(level)
    return _PRESETS.get(level, NORMAL_CONFIG)


def extract_with_quality(
    file_path: str,
    level: QualityLevel = QualityLevel.NORMAL,
    **overrides,
) -> Any:
    from runeextract import extract
    config = get_quality_config(level)
    kwargs = config.to_extract_kwargs()
    kwargs.update(overrides)
    return extract(file_path, **kwargs)


def run_pipeline_with_quality(
    file_paths: List[str],
    level: QualityLevel = QualityLevel.NORMAL,
    **overrides,
) -> List[Any]:
    config = get_quality_config(level)
    kwargs = config.to_extract_kwargs()
    kwargs.update(overrides)
    from runeextract import extract_many
    return extract_many(file_paths, **kwargs)


# ── Legacy compatibility API (replaces old runeextract/quality.py) ──────────


_LEGACY_MAP = {
    "fast": QualityLevel.DRAFT,
    "standard": QualityLevel.NORMAL,
    "high": QualityLevel.VERIFIED,
    "maximum": QualityLevel.VERIFIED,
}


def _resolve_level(level) -> QualityLevel:
    if isinstance(level, QualityLevel):
        return level
    if isinstance(level, str):
        mapped = _LEGACY_MAP.get(level.lower())
        if mapped:
            return mapped
        return QualityLevel(level.lower())
    return QualityLevel.NORMAL


@dataclass
class FastMode:
    """Legacy compatibility wrapper — use ``QualityConfig`` instead."""

    enabled: bool = False
    skip_ocr: bool = True
    skip_images: bool = True
    skip_tables: bool = False
    skip_metadata: bool = False
    skip_chunking: bool = True
    skip_embeddings: bool = True
    max_pages: Optional[int] = None
    extract_timeout: int = 30
    _level: QualityLevel = QualityLevel.NORMAL

    @classmethod
    def for_level(cls, level: QualityLevel) -> "FastMode":
        resolved = _resolve_level(level)
        config = get_quality_config(resolved)
        skip_ocr = not config.ocr
        skip_images = not config.images
        skip_tables = not config.tables
        skip_metadata = not config.metadata
        timeout_map = {QualityLevel.DRAFT: 15, QualityLevel.NORMAL: 60, QualityLevel.VERIFIED: 300}
        return cls(
            enabled=resolved == QualityLevel.DRAFT,
            skip_ocr=skip_ocr,
            skip_images=skip_images,
            skip_tables=skip_tables,
            skip_metadata=skip_metadata,
            skip_chunking=not config.chunking,
            skip_embeddings=not config.ai_analysis,
            max_pages=5 if resolved == QualityLevel.DRAFT else None,
            extract_timeout=timeout_map.get(resolved, 60),
            _level=resolved,
        )


def configure_quality(level=None, **kwargs) -> FastMode:
    if level is None and "level" in kwargs:
        level = kwargs.pop("level")
    if level is None:
        level = QualityLevel.NORMAL
    resolved = _resolve_level(level)
    return FastMode.for_level(resolved)
