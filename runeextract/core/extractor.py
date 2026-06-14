"""
Base extractor class and main extraction interface.
"""

import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List, AsyncIterator
from runeextract.models.document import Document
from runeextract.exceptions import (
    UnsupportedFormatError, FileTooLargeError, CorruptFileError, ExtractionError
)

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """
    Abstract base class for all document extractors.
    
    All extractors must inherit from this class and implement the extract method.
    This ensures consistent output schema across all file types.
    """
    
    def __init__(self, ocr: bool = False, **kwargs):
        """
        Initialize the extractor.
        
        Args:
            ocr: Whether to use OCR for text extraction
            **kwargs: Additional extractor-specific options
        """
        self.ocr = ocr
        self.options = kwargs
        self.max_file_size = kwargs.get("max_file_size", 500 * 1024 * 1024)
    
    @abstractmethod
    def extract(self, file_path: str) -> Document:
        """
        Extract content from a file.
        
        Args:
            file_path: Path to the file to extract
            
        Returns:
            Document object with extracted content
        """
        pass
    
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """
        Return list of supported file extensions.
        
        Returns:
            List of file extensions (e.g., [".pdf", ".PDF"])
        """
        pass
    
    def validate_file(self, file_path: str) -> bool:
        """
        Validate that the file exists and is supported.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file is valid
            
        Raises:
            FileNotFoundError: If file doesn't exist
            UnsupportedFormatError: If file type not supported
            FileTooLargeError: If file exceeds size limit
            CorruptFileError: If file is empty
            ExtractionError: If read permission is denied or path is a directory
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not path.is_file():
            raise ExtractionError(
                f"Path is not a file: {file_path}",
                file_path=file_path,
                error_code="E041"
            )
        
        if not os.access(file_path, os.R_OK):
            raise ExtractionError(
                f"No read permission: {file_path}",
                file_path=file_path,
                error_code="E040"
            )
        
        try:
            stat = path.stat()
        except OSError as exc:
            raise ExtractionError(
                f"Cannot access file: {file_path} - {exc}",
                file_path=file_path,
                error_code="E042"
            )
        
        file_size = stat.st_size
        if file_size == 0:
            raise CorruptFileError(file_path, detail="File is empty (0 bytes)")
        if file_size > self.max_file_size:
            raise FileTooLargeError(file_path, file_size, self.max_file_size)
        
        extension = path.suffix.lower()
        if extension not in [ext.lower() for ext in self.supported_extensions()]:
            raise UnsupportedFormatError(file_path, extension=extension)
        
        return True
    
    def clean_text(self, text: str, preserve_whitespace: bool = False) -> str:
        """
        Clean extracted text.
        
        Args:
            text: Raw text to clean
            preserve_whitespace: If True, preserve all whitespace (safe for code blocks).
                                 If False (default), collapse runs of spaces and excess newlines.
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        if not preserve_whitespace:
            text = re.sub(r'[ \t]+', ' ', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()


class StreamingExtractor(BaseExtractor):
    """
    Extractor that supports page/section streaming.

    Implement extract_stream to yield partial Document objects
    (e.g., one per page) as they become available.
    """

    async def extract_stream(self, file_path: str) -> AsyncIterator[Document]:
        """
        Extract content progressively, yielding partial Documents.

        Default implementation yields the full document in one chunk.
        Override for page-by-page streaming.
        """
        yield self.extract(file_path)
