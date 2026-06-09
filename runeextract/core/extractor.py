"""
Base extractor class and main extraction interface.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
from runeextract.models.document import Document


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
            True if file is valid, False otherwise
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        extension = path.suffix.lower()
        if extension not in [ext.lower() for ext in self.supported_extensions()]:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported: {self.supported_extensions()}"
            )
        
        return True
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        return text.strip()
