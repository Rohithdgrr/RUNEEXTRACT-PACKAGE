"""
Plugin registry for custom extractors.
"""

import logging
import threading
from typing import Dict, Type, List
from runeextract.core.extractor import BaseExtractor

logger = logging.getLogger(__name__)


class ExtractorRegistry:
    """
    Registry for document extractors.
    
    Allows registration of custom extractors for different file types.
    """
    
    _instance = None
    _extractors: Dict[str, Type[BaseExtractor]] = {}
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register(cls, extension: str):
        """
        Decorator to register an extractor for a file extension.
        
        Args:
            extension: File extension (e.g., ".pdf")
            
        Example:
            @ExtractorRegistry.register(".pdf")
            class PDFExtractor(BaseExtractor):
                ...
        """
        def decorator(extractor_class: Type[BaseExtractor]):
            with cls._lock:
                cls._extractors[extension.lower()] = extractor_class
            return extractor_class
        return decorator
    
    @classmethod
    def get_extractor(cls, extension: str) -> Type[BaseExtractor]:
        """
        Get extractor class for a file extension.
        
        Args:
            extension: File extension (e.g., ".pdf")
            
        Returns:
            Extractor class
            
        Raises:
            KeyError: If no extractor is registered for the extension
        """
        extension = extension.lower()
        if extension not in cls._extractors:
            raise KeyError(f"No extractor registered for extension: {extension}")
        return cls._extractors[extension]
    
    @classmethod
    def is_registered(cls, extension: str) -> bool:
        """
        Check if an extractor is registered for an extension.
        
        Args:
            extension: File extension
            
        Returns:
            True if registered, False otherwise
        """
        return extension.lower() in cls._extractors
    
    @classmethod
    def registered_extensions(cls) -> List[str]:
        """
        Get list of all registered extensions.
        
        Returns:
            List of registered file extensions
        """
        return list(cls._extractors.keys())
    
    @classmethod
    def unregister(cls, extension: str) -> bool:
        """
        Unregister an extractor for an extension.
        
        Args:
            extension: File extension
            
        Returns:
            True if unregistered, False if not found
        """
        extension = extension.lower()
        with cls._lock:
            if extension in cls._extractors:
                del cls._extractors[extension]
                return True
        return False

    @classmethod
    def discover(cls, group: str = "runeextract.extractors") -> int:
        """
        Discover and register extractors via importlib.metadata entry points.
        
        Args:
            group: Entry point group name
            
        Returns:
            Number of extractors discovered
        """
        try:
            from importlib import metadata as importlib_metadata
        except ImportError:
            return 0
        
        count = 0
        for entry_point in importlib_metadata.entry_points(group=group):
            try:
                extractor_class = entry_point.load()
                if hasattr(extractor_class, "supported_extensions"):
                    for ext in extractor_class.supported_extensions():
                        with cls._lock:
                            cls._extractors[ext.lower()] = extractor_class
                        count += 1
                    logger.debug(f"Discovered extractor {entry_point.name}: {extractor_class.__name__}")
            except Exception as exc:
                logger.warning(f"Failed to load extractor {entry_point.name}: {exc}")
        return count


# Convenience function for decorator
def register_extractor(extension: str):
    """
    Convenience function to register an extractor.
    
    Args:
        extension: File extension (e.g., ".pdf")
        
    Example:
        @register_extractor(".epub")
        class EPUBExtractor(BaseExtractor):
            ...
    """
    return ExtractorRegistry.register(extension)
