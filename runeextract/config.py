"""
Configuration system for RuneExtract.

Supports environment variables, JSON config files, and pyproject.toml.
"""

import os
import json
import logging
from dataclasses import dataclass, field, fields
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Env var prefixes
ENV_PREFIX = "RUNEEXTRACT_"


def _env(key: str, default: Any = None) -> Any:
    """Read an env var with the RUNEEXTRACT_ prefix."""
    return os.environ.get(f"{ENV_PREFIX}{key}", default)


def _parse_bool(val: Any, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("1", "true", "yes", "on")
    return default


def _parse_int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


@dataclass
class RuneExtractConfig:
    """
    Global configuration for RuneExtract.

    Resolved priority (highest wins):
      1. Per-call kwargs
      2. Environment variables (RUNEEXTRACT_*)
      3. JSON config file (~/.runeextract.json)
      4. pyproject.toml [tool.runeextract]
      5. Defaults
    """
    ocr: bool = False
    tables: bool = True
    images: bool = True
    metadata: bool = True
    chunking_strategy: Optional[str] = None
    chunk_size: int = 1000
    chunk_overlap: int = 100
    max_file_size: int = 500 * 1024 * 1024  # 500 MB
    allowed_extensions: Optional[List[str]] = None
    cache_dir: Optional[str] = None
    cache_ttl: int = 3600
    log_level: str = "WARNING"
    extra: Dict[str, Any] = field(default_factory=dict)
    _skip_init: bool = field(default=False, repr=False, init=False)

    def __post_init__(self):
        if self._skip_init:
            return
        # Apply lowest priority first, highest priority last (last write wins)
        self.apply_pyproject()
        self.apply_config_file()
        self.apply_env()

    def apply_env(self):
        """Override settings from environment variables."""
        raw = {}
        for key in (
            "OCR", "TABLES", "IMAGES", "METADATA",
            "CHUNKING_STRATEGY", "CHUNK_SIZE", "CHUNK_OVERLAP",
            "MAX_FILE_SIZE", "CACHE_DIR", "CACHE_TTL", "LOG_LEVEL"
        ):
            val = _env(key)
            if val is not None:
                raw[key.lower()] = val

        if "ocr" in raw:
            self.ocr = _parse_bool(raw["ocr"], self.ocr)
        if "tables" in raw:
            self.tables = _parse_bool(raw["tables"], self.tables)
        if "images" in raw:
            self.images = _parse_bool(raw["images"], self.images)
        if "metadata" in raw:
            self.metadata = _parse_bool(raw["metadata"], self.metadata)
        if "chunking_strategy" in raw:
            self.chunking_strategy = str(raw["chunking_strategy"]) or None
        if "chunk_size" in raw:
            self.chunk_size = _parse_int(raw["chunk_size"], self.chunk_size)
        if "chunk_overlap" in raw:
            self.chunk_overlap = _parse_int(raw["chunk_overlap"], self.chunk_overlap)
        if "max_file_size" in raw:
            self.max_file_size = _parse_int(raw["max_file_size"], self.max_file_size)
        if "cache_dir" in raw:
            self.cache_dir = str(raw["cache_dir"])
        if "cache_ttl" in raw:
            self.cache_ttl = _parse_int(raw["cache_ttl"], self.cache_ttl)
        if "log_level" in raw:
            self.log_level = str(raw["log_level"]).upper()

    def apply_config_file(self):
        """Load settings from ~/.runeextract.json."""
        paths = [
            Path.home() / ".runeextract.json",
            Path.cwd() / ".runeextract.json",
        ]
        for path in paths:
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    self._merge_dict(data)
                    logger.debug(f"Loaded config from {path}")
                except Exception as exc:
                    logger.warning(f"Failed to load config {path}: {exc}")

    def apply_pyproject(self):
        """Load settings from pyproject.toml [tool.runeextract]."""
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                logger.debug("tomllib/tomli not available; skipping pyproject.toml config")
                return

        paths = [Path.cwd() / "pyproject.toml"]
        for path in paths:
            if path.exists():
                try:
                    with open(path, "rb") as fh:
                        data = tomllib.load(fh)
                    section = data.get("tool", {}).get("runeextract", {})
                    if section:
                        self._merge_dict(section)
                        logger.debug(f"Loaded config from {path}")
                except Exception as exc:
                    logger.warning(f"Failed to load pyproject config from {path}: {exc}")

    _BOOL_FIELDS = {"ocr", "tables", "images", "metadata"}
    _INT_FIELDS = {"chunk_size", "chunk_overlap", "max_file_size", "cache_ttl"}

    def _coerce(self, field_name: str, value: Any) -> Any:
        if field_name in self._BOOL_FIELDS:
            return _parse_bool(value)
        if field_name in self._INT_FIELDS:
            return _parse_int(value)
        return value

    def _merge_dict(self, data: dict):
        """Merge a dict into this config, ignoring unknown keys."""
        mapping = {k.lower(): v for k, v in data.items()}
        for field_name in (
            "ocr", "tables", "images", "metadata",
            "chunking_strategy", "chunk_size", "chunk_overlap",
            "max_file_size", "allowed_extensions", "cache_dir",
            "cache_ttl", "log_level"
        ):
            if field_name in mapping:
                setattr(self, field_name, self._coerce(field_name, mapping[field_name]))
        # Collect extra keys
        known = {
            "ocr", "tables", "images", "metadata",
            "chunking_strategy", "chunk_size", "chunk_overlap",
            "max_file_size", "allowed_extensions", "cache_dir",
            "cache_ttl", "log_level"
        }
        for k, v in mapping.items():
            if k not in known:
                self.extra[k] = v

    def merge_options(self, **kwargs) -> "RuneExtractConfig":
        """Return a new config with per-call kwargs overlaid (avoids __post_init__ re-apply)."""
        # Bypass __init__ to prevent __post_init__ from re-applying env/config/pyproject
        clone = object.__new__(RuneExtractConfig)
        # Copy all fields from self
        for f in fields(self):
            setattr(clone, f.name, getattr(self, f.name))
        # Override with kwargs
        for k, v in kwargs.items():
            if hasattr(clone, k):
                setattr(clone, k, v)
            else:
                clone.extra[k] = v
        clone._skip_init = True
        return clone

    def to_dict(self) -> dict:
        return {
            "ocr": self.ocr,
            "tables": self.tables,
            "images": self.images,
            "metadata": self.metadata,
            "chunking_strategy": self.chunking_strategy,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "max_file_size": self.max_file_size,
            "allowed_extensions": self.allowed_extensions,
            "cache_dir": self.cache_dir,
            "cache_ttl": self.cache_ttl,
            "log_level": self.log_level,
        }


# Global singleton
_default_config: Optional[RuneExtractConfig] = None


def get_config() -> RuneExtractConfig:
    """Get or create the global default configuration."""
    global _default_config
    if _default_config is None:
        _default_config = RuneExtractConfig()
    return _default_config


def set_config(cfg: RuneExtractConfig):
    """Override the global default configuration."""
    global _default_config
    _default_config = cfg
