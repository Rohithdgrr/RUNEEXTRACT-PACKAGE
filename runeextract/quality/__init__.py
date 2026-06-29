"""Quality levels / fast mode — processing quality presets (draft/normal/verified).

Usage:
    from runeextract.quality import QualityLevel, extract_with_quality

    doc = extract_with_quality("file.pdf", level=QualityLevel.DRAFT)
"""

from runeextract.quality.levels import (
    QualityLevel,
    QualityConfig,
    extract_with_quality,
    get_quality_config,
    run_pipeline_with_quality,
    resolution_map,
    FastMode,
    configure_quality,
)

__all__ = [
    "QualityLevel",
    "QualityConfig",
    "extract_with_quality",
    "get_quality_config",
    "run_pipeline_with_quality",
    "resolution_map",
    "FastMode",
    "configure_quality",
]
