"""
Schema definitions and validation.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class ExtractionOptions:
    """Options for document extraction."""
    ocr: bool = False
    tables: bool = True
    images: bool = True
    metadata: bool = True
    chunking_strategy: Optional[str] = None
    chunk_size: int = 1000
    chunk_overlap: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'ocr': self.ocr,
            'tables': self.tables,
            'images': self.images,
            'metadata': self.metadata,
            'chunking_strategy': self.chunking_strategy,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }


@dataclass
class ExtractionResult:
    """Result of document extraction."""
    success: bool
    document: Optional[Any] = None
    error: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
