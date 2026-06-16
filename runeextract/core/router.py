"""
File type router for selecting appropriate extractor.
"""

import logging
import os
from pathlib import Path
from typing import Optional
from runeextract.core.extractor import BaseExtractor
from runeextract.core.registry import ExtractorRegistry
from runeextract.exceptions import UnsupportedFormatError

logger = logging.getLogger(__name__)

# URL-based extractors: (detect function, module path)
_URL_EXTRACTORS = [
    ("runeextract.extractors.youtube.extractor.YoutubeExtractor", None),
    ("runeextract.extractors.notion.extractor.NotionExtractor", None),
]


def _detect_youtube(file_path: str) -> bool:
    import re
    return bool(re.search(
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/shorts/)',
        file_path
    ))


def _detect_notion(file_path: str) -> bool:
    return "notion." in file_path or file_path.startswith("notion://") or (
        len(file_path.replace("-", "").replace("/", "")) >= 32
        and file_path.startswith("http")
        and "notion" in file_path.lower()
    )


_URL_EXTRACTORS = [
    (_detect_youtube, "runeextract.extractors.youtube.extractor.YoutubeExtractor"),
    (_detect_notion, "runeextract.extractors.notion.extractor.NotionExtractor"),
]


# Magic-bytes for content-based format detection (first 8 bytes)
_MAGIC_BYTES: dict[bytes, str] = {
    b'%PDF': '.pdf',
    b'PK\x03\x04': '.docx',       # ZIP-based (DOCX/XLSX/PPTX — check further)
    b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1': '.doc',  # OLE2 (old .doc/.ppt/.xls)
    b'\x89PNG\r\n\x1a\n': '.png',
    b'\xff\xd8\xff': '.jpg',
    b'II*\x00': '.tiff',           # TIFF little-endian
    b'MM\x00*': '.tiff',           # TIFF big-endian
    b'GIF87a': '.gif',
    b'GIF89a': '.gif',
    b'RIFF': '.webp',              # WebP (RIFF + WEBP)
    b'<!DOCTYPE html': '.html',
    b'<html': '.html',
    b'<svg': '.svg',
    b'PK\x03\x04': '.zip',         # Generic ZIP fallback
}


def _detect_by_magic(file_path: str) -> Optional[str]:
    """Detect file extension from magic bytes. Returns None if unknown."""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(8)
    except OSError:
        return None

    # Check for PDF
    if header.startswith(b'%PDF'):
        return '.pdf'

    # Check for PNG
    if header.startswith(b'\x89PNG'):
        return '.png'

    # Check for JPEG (starts with FF D8 FF)
    if header[:3] == b'\xff\xd8\xff':
        return '.jpg'

    # Check for TIFF (II = little-endian, MM = big-endian)
    if header[:2] in (b'II', b'MM') and header[2:4] in (b'\x2a\x00', b'\x00\x2a'):
        return '.tiff'

    # Check for GIF
    if header[:3] in (b'GIF',):
        return '.gif'

    # Check for ZIP-based formats (DOCX, XLSX, PPTX, ZIP)
    if header[:2] == b'PK':
        try:
            import zipfile
            with zipfile.ZipFile(file_path) as z:
                names = z.namelist()
                if 'word/document.xml' in names:
                    return '.docx'
                if 'xl/workbook.xml' in names:
                    return '.xlsx'
                if 'ppt/presentation.xml' in names:
                    return '.pptx'
        except Exception:
            pass
        return '.zip'

    # Check for OLE2 (old .doc/.ppt/.xls)
    if header[:8] == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
        return '.doc'

    # Check for HTML
    stripped = header.lstrip(b'\xef\xbb\xbf\xfe\xff ')  # skip BOM + leading space
    if stripped.lower().startswith(b'<!doctype html') or stripped.lower().startswith(b'<html'):
        return '.html'

    return None


class ExtractorRouter:
    """
    Routes files to the appropriate extractor based on file type.
    """
    
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
        '.csv': 'runeextract.extractors.csv.extractor.CsvExtractor',
        '.json': 'runeextract.extractors.json.extractor.JsonExtractor',
        '.png': 'runeextract.extractors.image.extractor.ImageExtractor',
        '.jpg': 'runeextract.extractors.image.extractor.ImageExtractor',
        '.jpeg': 'runeextract.extractors.image.extractor.ImageExtractor',
        '.tiff': 'runeextract.extractors.image.extractor.ImageExtractor',
        '.tif': 'runeextract.extractors.image.extractor.ImageExtractor',
        '.bmp': 'runeextract.extractors.image.extractor.ImageExtractor',
        '.webp': 'runeextract.extractors.image.extractor.ImageExtractor',
        '.epub': 'runeextract.extractors.epub.extractor.EpubExtractor',
        '.mp3': 'runeextract.extractors.audio.extractor.AudioExtractor',
        '.wav': 'runeextract.extractors.audio.extractor.AudioExtractor',
        '.flac': 'runeextract.extractors.audio.extractor.AudioExtractor',
        '.m4a': 'runeextract.extractors.audio.extractor.AudioExtractor',
        '.ogg': 'runeextract.extractors.audio.extractor.AudioExtractor',
        '.wma': 'runeextract.extractors.audio.extractor.AudioExtractor',
        '.aac': 'runeextract.extractors.audio.extractor.AudioExtractor',
        '.opus': 'runeextract.extractors.audio.extractor.AudioExtractor',
        '.mp4': 'runeextract.extractors.video.extractor.VideoExtractor',
        '.avi': 'runeextract.extractors.video.extractor.VideoExtractor',
        '.mov': 'runeextract.extractors.video.extractor.VideoExtractor',
        '.mkv': 'runeextract.extractors.video.extractor.VideoExtractor',
        '.webm': 'runeextract.extractors.video.extractor.VideoExtractor',
        '.flv': 'runeextract.extractors.video.extractor.VideoExtractor',
        '.wmv': 'runeextract.extractors.video.extractor.VideoExtractor',
    }
    
    @classmethod
    def get_extractor(cls, file_path: str, **kwargs) -> BaseExtractor:
        """
        Get the appropriate extractor for a file or URL.
        
        Args:
            file_path: Path to the file or URL
            **kwargs: Options to pass to the extractor
            
        Returns:
            Instantiated extractor
            
        Raises:
            UnsupportedFormatError: If no extractor is available
        """
        # Check URL-based extractors first
        for detector, module_path in _URL_EXTRACTORS:
            if detector(file_path):
                try:
                    return cls._import_extractor(module_path, **kwargs)
                except ImportError as exc:
                    logger.warning(f"URL extractor {module_path} unavailable: {exc}")
                break
        
        path = Path(file_path)
        extension = path.suffix.lower()
        
        # Check registry first (custom extractors)
        if extension and ExtractorRegistry.is_registered(extension):
            extractor_class = ExtractorRegistry.get_extractor(extension)
            return extractor_class(**kwargs)
        
        # Fall back to built-in extractors
        if extension in cls.BUILTIN_EXTRACTORS:
            module_path = cls.BUILTIN_EXTRACTORS[extension]
            return cls._import_extractor(module_path, **kwargs)
        
        # Content-based detection via magic bytes
        if os.path.isfile(file_path):
            magic_ext = _detect_by_magic(file_path)
            if magic_ext:
                if magic_ext in cls.BUILTIN_EXTRACTORS:
                    logger.debug(f"Resolved {file_path} -> {magic_ext} via magic bytes")
                    return cls._import_extractor(cls.BUILTIN_EXTRACTORS[magic_ext], **kwargs)
                if ExtractorRegistry.is_registered(magic_ext):
                    return ExtractorRegistry.get_extractor(magic_ext)(**kwargs)
        
        raise UnsupportedFormatError(file_path, extension=extension)
    
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
            Source type string (e.g., "pdf", "docx", "html", "markdown")
        """
        path = Path(file_path)
        ext = path.suffix.lower().lstrip('.')
        # Map extensions to canonical names
        ext_map = {
            'md': 'markdown',
            'markdown': 'markdown',
            'doc': 'docx',
            'docx': 'docx',
            'ppt': 'pptx',
            'pptx': 'pptx',
            'xls': 'xlsx',
            'xlsx': 'xlsx',
            'htm': 'html',
            'html': 'html',
            'csv': 'csv',
            'json': 'json',
            'png': 'image',
            'jpg': 'image',
            'jpeg': 'image',
            'tiff': 'image',
            'tif': 'image',
            'bmp': 'image',
            'webp': 'image',
            'epub': 'epub',
            'mp3': 'audio',
            'wav': 'audio',
            'flac': 'audio',
            'm4a': 'audio',
            'ogg': 'audio',
            'wma': 'audio',
            'aac': 'audio',
            'opus': 'audio',
            'mp4': 'video',
            'avi': 'video',
            'mov': 'video',
            'mkv': 'video',
            'webm': 'video',
            'flv': 'video',
            'wmv': 'video',
        }
        return ext_map.get(ext, ext)
