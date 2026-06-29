from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

from runeextract.exceptions import ImageSizeError

_MAX_IMAGE_SIZE = 50 * 1024 * 1024

logger = logging.getLogger(__name__)

_TOKEN_ENCODING_CACHE = {}


def _get_token_encoding(encoding_name: str = "cl100k_base"):
    if encoding_name not in _TOKEN_ENCODING_CACHE:
        try:
            import tiktoken
            _TOKEN_ENCODING_CACHE[encoding_name] = tiktoken.get_encoding(encoding_name)
        except ImportError:
            _TOKEN_ENCODING_CACHE[encoding_name] = None
    return _TOKEN_ENCODING_CACHE.get(encoding_name)


def _estimate_token_count(text: str) -> int:
    return max(1, len(text) // 4)


class ChunkingStrategy(str, Enum):
    BY_PAGE = "by_page"
    BY_HEADING = "by_heading"
    SEMANTIC = "semantic"
    FIXED_SIZE = "fixed_size"
    BY_TOKEN = "by_token"
    SENTENCE_WINDOW = "sentence_window"
    HIERARCHICAL = "hierarchical"


@dataclass
class Image:
    data: bytes
    format: str
    width: Optional[int] = None
    height: Optional[int] = None
    page_number: Optional[int] = None
    caption: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.data) > _MAX_IMAGE_SIZE:
            raise ImageSizeError(len(self.data), _MAX_IMAGE_SIZE)


@dataclass
class Table:
    rows: List[List[str]]
    columns: List[str]
    page_number: Optional[int] = None
    caption: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self):
        try:
            import pandas as pd
            return pd.DataFrame(self.rows, columns=self.columns)
        except ImportError:
            raise ImportError("pandas is required for DataFrame conversion. Install with: pip install pandas")


@dataclass
class Chunk:
    text: str
    chunk_id: str
    start_index: int
    end_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_document_id: Optional[str] = None
    parent_chunk_id: Optional[str] = None

    def token_count(self, encoding_name: str = "cl100k_base") -> int:
        enc = _get_token_encoding(encoding_name)
        if enc:
            return len(enc.encode(self.text))
        return _estimate_token_count(self.text)

    def is_child(self) -> bool:
        """Whether this chunk has a parent chunk (level 0 in hierarchy)."""
        return self.parent_chunk_id is not None

    def is_parent(self) -> bool:
        """Whether this chunk serves as a parent in a hierarchy (level 1)."""
        return self.metadata.get("level") == 1


@dataclass
class HierarchicalChunk(Chunk):
    """A chunk that belongs to a parent-child hierarchy.

    Extends Chunk with level metadata and a reference to children.
    """
    level: int = 0
    children: List[str] = field(default_factory=list)

    def token_count(self, encoding_name: str = "cl100k_base") -> int:
        enc = _get_token_encoding(encoding_name)
        if enc:
            return len(enc.encode(self.text))
        return _estimate_token_count(self.text)
