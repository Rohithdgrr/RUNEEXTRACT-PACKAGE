"""
Plugin registry for custom extractors.
"""

from typing import Dict, Type, List
from runeextract.core.extractor import BaseExtractor


class ExtractorRegistry:
    """
    Registry for document extractors.
    
    Allows registration of custom extractors for different file types.
    """
    
    _instance = None
    _extractors: Dict[str, Type[BaseExtractor]] = {}
    
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
        if extension in cls._extractors:
            del cls._extractors[extension]
            return True
        return False


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
