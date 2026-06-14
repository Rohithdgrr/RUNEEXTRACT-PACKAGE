"""
Document model for unified extraction output.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum


class ChunkingStrategy(str, Enum):
    """Chunking strategies for document processing."""
    BY_PAGE = "by_page"
    BY_HEADING = "by_heading"
    SEMANTIC = "semantic"
    FIXED_SIZE = "fixed_size"


@dataclass
class Image:
    """Represents an extracted image."""
    data: bytes
    format: str
    width: Optional[int] = None
    height: Optional[int] = None
    page_number: Optional[int] = None
    caption: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Table:
    """Represents an extracted table."""
    rows: List[List[str]]
    columns: List[str]
    page_number: Optional[int] = None
    caption: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self):
        """Convert table to pandas DataFrame."""
        try:
            import pandas as pd
            return pd.DataFrame(self.rows, columns=self.columns)
        except ImportError:
            raise ImportError("pandas is required for DataFrame conversion. Install with: pip install pandas")


@dataclass
class Chunk:
    """Represents a chunk of text for RAG applications."""
    text: str
    chunk_id: str
    start_index: int
    end_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """
    Universal document model for all extraction types.
    
    No matter the source (PDF, DOCX, HTML, etc.), the output schema is identical.
    """
    text: str
    tables: List[Table] = field(default_factory=list)
    images: List[Image] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_type: str = ""
    source_path: Optional[str] = None
    _chunks: Optional[List[Chunk]] = field(default=None, repr=False)

    def chunks(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE,
        size: int = 1000,
        overlap: int = 100,
        **kwargs
    ) -> List[Chunk]:
        """
        Chunk the document text for RAG applications.
        
        Args:
            strategy: Chunking strategy (by_page, by_heading, semantic, fixed_size)
            size: Target chunk size in characters
            overlap: Character overlap between chunks
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            List of Chunk objects
        """
        if self._chunks is not None:
            return self._chunks

        if strategy == ChunkingStrategy.FIXED_SIZE:
            self._chunks = self._chunk_fixed_size(size, overlap)
        elif strategy == ChunkingStrategy.BY_PAGE:
            self._chunks = self._chunk_by_page(**kwargs)
        elif strategy == ChunkingStrategy.BY_HEADING:
            self._chunks = self._chunk_by_heading(**kwargs)
        elif strategy == ChunkingStrategy.SEMANTIC:
            self._chunks = self._chunk_semantic(size, **kwargs)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        return self._chunks

    def _chunk_fixed_size(self, size: int, overlap: int) -> List[Chunk]:
        """Chunk text by fixed size with overlap."""
        chunks = []
        text = self.text
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + size
            chunk_text = text[start:end]
            
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=f"chunk_{chunk_id}",
                start_index=start,
                end_index=end,
                metadata={"strategy": "fixed_size", "size": size, "overlap": overlap}
            ))
            
            step = max(size - overlap, 1)
            start += step
            chunk_id += 1

        return chunks

    def _chunk_by_page(self, **kwargs) -> List[Chunk]:
        """Chunk text by page boundaries."""
        # This would require page information from metadata
        # For now, return as single chunk
        return [Chunk(
            text=self.text,
            chunk_id="chunk_0",
            start_index=0,
            end_index=len(self.text),
            metadata={"strategy": "by_page"}
        )]

    def _chunk_by_heading(self, **kwargs) -> List[Chunk]:
        """Chunk text by heading structure."""
        # This would require heading information from metadata
        # For now, return as single chunk
        return [Chunk(
            text=self.text,
            chunk_id="chunk_0",
            start_index=0,
            end_index=len(self.text),
            metadata={"strategy": "by_heading"}
        )]

    def _chunk_semantic(self, size: int, **kwargs) -> List[Chunk]:
        """Chunk text using semantic boundaries (sentences, paragraphs)."""
        # Simple implementation: split by paragraphs
        paragraphs = self.text.split('\n\n')
        chunks = []
        current_chunk = ""
        chunk_id = 0
        start_index = 0

        for para in paragraphs:
            if len(current_chunk) + len(para) > size and current_chunk:
                chunks.append(Chunk(
                    text=current_chunk.strip(),
                    chunk_id=f"chunk_{chunk_id}",
                    start_index=start_index,
                    end_index=start_index + len(current_chunk),
                    metadata={"strategy": "semantic"}
                ))
                start_index += len(current_chunk)
                current_chunk = para
                chunk_id += 1
            else:
                current_chunk += "\n\n" + para if current_chunk else para

        if current_chunk:
            chunks.append(Chunk(
                text=current_chunk.strip(),
                chunk_id=f"chunk_{chunk_id}",
                start_index=start_index,
                end_index=start_index + len(current_chunk),
                metadata={"strategy": "semantic"}
            ))

        return chunks

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary representation."""
        return {
            "text": self.text,
            "tables": [
                {
                    "rows": table.rows,
                    "columns": table.columns,
                    "page_number": table.page_number,
                    "caption": table.caption,
                    "metadata": table.metadata
                }
                for table in self.tables
            ],
            "images": [
                {
                    "format": img.format,
                    "width": img.width,
                    "height": img.height,
                    "page_number": img.page_number,
                    "caption": img.caption,
                    "metadata": img.metadata,
                    "data_size": len(img.data)
                }
                for img in self.images
            ],
            "metadata": self.metadata,
            "source_type": self.source_type,
            "source_path": self.source_path,
            "chunks": [
                {
                    "text": chunk.text,
                    "chunk_id": chunk.chunk_id,
                    "start_index": chunk.start_index,
                    "end_index": chunk.end_index,
                    "metadata": chunk.metadata
                }
                for chunk in (self._chunks or [])
            ]
        }
