"""
File type router for selecting appropriate extractor.
"""

from pathlib import Path
from typing import Optional
from runeextract.core.extractor import BaseExtractor
from runeextract.core.registry import ExtractorRegistry


class ExtractorRouter:
    """
    Routes files to the appropriate extractor based on file type.
    """
    
    # Built-in extractor mappings
    BUILTIN_EXTRACTORS = {
        '.pdf': 'runeextract.extractors.pdf.extractor.PDFExtractor',
        '.docx': 'runeextract.extractors.docx.extractor.DocxExtractor',
        '.doc': 'runeextract.extractors.docx.extractor.DocxExtractor',
        '.pptx': 'runeextract.extractors.pptx.extractor.PptxExtractor',
        '.ppt': 'runeextract.extractors.pptx.extractor.PptxExtractor',
        '.xlsx': 'runeextract.extractors.xlsx.extractor.XlsxExtractor',
        '.xls': 'runeextract.extractors.xlsx.extractor.XlsxExtractor',
        '.html': 'runeextract.extractors.html.extractor.HtmlExtractor',
        '.htm': 'runeextract.extractors.html.extractor.HtmlExtractor',
        '.md': 'runeextract.extractors.markdown.extractor.MarkdownExtractor',
        '.markdown': 'runeextract.extractors.markdown.extractor.MarkdownExtractor',
    }
    
    @classmethod
    def get_extractor(cls, file_path: str, **kwargs) -> BaseExtractor:
        """
        Get the appropriate extractor for a file.
        
        Args:
            file_path: Path to the file
            **kwargs: Options to pass to the extractor
            
        Returns:
            Instantiated extractor
            
        Raises:
            ValueError: If no extractor is available for the file type
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        # Check registry first (custom extractors)
        if ExtractorRegistry.is_registered(extension):
            extractor_class = ExtractorRegistry.get_extractor(extension)
            return extractor_class(**kwargs)
        
        # Fall back to built-in extractors
        if extension in cls.BUILTIN_EXTRACTORS:
            module_path = cls.BUILTIN_EXTRACTORS[extension]
            return cls._import_extractor(module_path, **kwargs)
        
        raise ValueError(
            f"No extractor available for file type: {extension}. "
            f"Supported types: {cls.supported_extensions()}"
        )
    
    @classmethod
    def _import_extractor(cls, module_path: str, **kwargs) -> BaseExtractor:
        """
        Dynamically import and instantiate an extractor.
        
        Args:
            module_path: Full module path to extractor class
            **kwargs: Options to pass to the extractor
            
        Returns:
            Instantiated extractor
        """
        try:
            module_name, class_name = module_path.rsplit('.', 1)
            module = __import__(module_name, fromlist=[class_name])
            extractor_class = getattr(module, class_name)
            return extractor_class(**kwargs)
        except ImportError as e:
            raise ImportError(
                f"Failed to import extractor for {module_path}: {e}. "
                f"Make sure the required dependencies are installed."
            )
    
    @classmethod
    def supported_extensions(cls) -> list[str]:
        """
        Get list of all supported file extensions.
        
        Returns:
            List of supported extensions
        """
        extensions = list(cls.BUILTIN_EXTRACTORS.keys())
        extensions.extend(ExtractorRegistry.registered_extensions())
        return list(set(extensions))
    
    @classmethod
    def get_source_type(cls, file_path: str) -> str:
        """
        Get the source type identifier for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Source type string (e.g., "pdf", "docx", "html")
        """
        path = Path(file_path)
        return path.suffix.lower().lstrip('.')
