"""
File type router for selecting appropriate extractor.
"""

import logging
import os
import ipaddress
from pathlib import Path
from typing import Optional, Dict
from urllib.parse import urlparse
from runeextract.core.extractor import BaseExtractor
from runeextract.core.registry import ExtractorRegistry
from runeextract.exceptions import UnsupportedFormatError, URLBlockedError, SSRFBlockedError, SecurityError, PathTraversalError, BombDetectionError
from runeextract.utils.logging import log_security_event

logger = logging.getLogger(__name__)

def _detect_youtube(file_path: str) -> bool:
    import re
    return bool(re.search(
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/shorts/)',
        file_path
    ))


def _detect_notion(file_path: str) -> bool:
    import re
    if file_path.startswith("notion://"):
        return True
    if re.search(r'notion\.(so|site)/', file_path, re.IGNORECASE):
        return True
    return False


_URL_EXTRACTORS = [
    (_detect_youtube, "runeextract.extractors.youtube.extractor.YoutubeExtractor"),
    (_detect_notion, "runeextract.extractors.notion.extractor.NotionExtractor"),
]


_MAX_ZIP_RATIO = 100  # max decompression ratio for zip bomb detection
_MAX_ZIP_FILES = 10000  # max entries in a zip archive


def _check_zip_bomb(file_path: str) -> None:
    """Detect zip bombs by checking compression ratio and entry count."""
    import zipfile
    try:
        with zipfile.ZipFile(file_path) as zf:
            info_list = zf.infolist()
            if len(info_list) > _MAX_ZIP_FILES:
                raise BombDetectionError(
                    f"Archive contains {len(info_list)} entries (max {_MAX_ZIP_FILES})",
                    file_path=file_path
                )
            compressed = sum(inf.compress_size for inf in info_list)
            uncompressed = sum(inf.file_size for inf in info_list)
            if compressed > 0 and uncompressed / compressed > _MAX_ZIP_RATIO:
                raise BombDetectionError(
                    f"Archive compression ratio {uncompressed // compressed}x exceeds limit {_MAX_ZIP_RATIO}x",
                    file_path=file_path
                )
    except (zipfile.BadZipFile, OSError) as e:
        logger.debug("Not a valid ZIP for bomb check: %s", e)


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
        except (zipfile.BadZipFile, OSError) as e:
            logger.debug("Not a known OOXML format: %s", e)
        return '.zip'

    # Check for OLE2 (old .doc/.ppt/.xls)
    if header[:8] == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
        return '.doc'

    # Check for HTML
    stripped = header.lstrip(b'\xef\xbb\xbf\xfe\xff ')  # skip BOM + leading space
    if stripped.lower().startswith(b'<!doctype html') or stripped.lower().startswith(b'<html'):
        return '.html'

    return None


class URLValidator:
    """Validate URLs against security policies to prevent SSRF and other attacks.

    Blocks private/internal IPs, localhost, and non-HTTP schemes.
    """

    ALLOWED_SCHEMES = {"http", "https"}
    ALLOWED_PORTS = {80, 443, 8080, 8443}
    BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "255.255.255.255"}

    @classmethod
    def validate(cls, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in cls.ALLOWED_SCHEMES:
            log_security_event("url_blocked", level="WARNING", url=url,
                               reason=f"scheme {parsed.scheme}", error_code="E101")
            raise URLBlockedError(url, reason=f"scheme '{parsed.scheme}' not allowed")
        hostname = parsed.hostname
        if not hostname:
            raise URLBlockedError(url, reason="no hostname")
        if hostname.lower() in cls.BLOCKED_HOSTS:
            log_security_event("ssrf_blocked", level="WARNING", url=url,
                               reason="localhost", error_code="E102")
            raise SSRFBlockedError(url)
        if parsed.port is not None and parsed.port not in cls.ALLOWED_PORTS:
            log_security_event("port_blocked", level="WARNING", url=url,
                               reason=f"port {parsed.port}", error_code="E101")
            raise URLBlockedError(url, reason=f"port {parsed.port} not allowed")
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_reserved:
                log_security_event("ssrf_blocked", level="WARNING", url=url,
                                   reason=f"private ip {hostname}", error_code="E102")
                raise SSRFBlockedError(url)
        except ValueError:
            cls._validate_dns(hostname, url)

    @classmethod
    def _validate_dns(cls, hostname: str, url: str) -> None:
        """Resolve hostname and check if it points to a private IP."""
        import socket
        try:
            resolved = socket.gethostbyname(hostname)
            ip = ipaddress.ip_address(resolved)
            if ip.is_private or ip.is_loopback or ip.is_reserved:
                log_security_event("ssrf_blocked", level="WARNING", url=url,
                                   reason=f"dns resolved to private {resolved}", error_code="E102")
                raise SSRFBlockedError(url)
        except OSError:
            pass

    @classmethod
    def validate_redirect_target(cls, url: str) -> str:
        """Validate URL after following a redirect (SSRF via redirect).

        Returns the final URL if safe, or None if validation fails.
        """
        import requests
        cls.validate(url)
        resp = requests.head(url, timeout=15, allow_redirects=True)
        final_url = resp.url
        if final_url != url:
            cls.validate(final_url)
        return final_url


def _check_path_traversal(file_path: str) -> None:
    """Detect path traversal attempts and null bytes in file paths.

    Raises PathTraversalError if the path is unsafe.
    """
    if "\x00" in file_path:
        raise PathTraversalError(file_path)

    if file_path.startswith("\\\\?\\") or file_path.startswith("\\\\.\\"):
        raise PathTraversalError(file_path)

    cleaned = file_path.replace("\\", "/")
    if cleaned == ".." or cleaned.startswith("../") or cleaned.endswith("/..") or "/../" in cleaned:
        raise PathTraversalError(file_path)

    if cleaned.startswith("//"):
        raise PathTraversalError(file_path)


def verify_file_type(file_path: str, expected_ext: str) -> bool:
    """Verify file content matches expected extension via magic bytes.

    Prevents extension-spoofing attacks. Returns True if the file's
    binary signature is consistent with its extension.

    Raises OSError if the file cannot be read (never silently returns True).
    """
    expected_ext = expected_ext.lstrip(".").lower()
    MAGIC = {
        "pdf": [b"%PDF"],
        "png": [b"\x89PNG\r\n\x1a\n"],
        "jpg": [b"\xff\xd8\xff"],
        "jpeg": [b"\xff\xd8\xff"],
        "gif": [b"GIF87a", b"GIF89a"],
        "webp": [b"RIFF"],
        "mp3": [b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"],
        "wav": [b"RIFF"],
    }
    ZIP_BASED = {"docx": "word/document.xml", "xlsx": "xl/workbook.xml",
                  "pptx": "ppt/presentation.xml", "epub": "mimetype"}

    with open(file_path, "rb") as f:
        header = f.read(32)

    sigs = MAGIC.get(expected_ext, [])
    if not sigs and expected_ext not in ZIP_BASED and expected_ext != "html":
        return True

    for sig in sigs:
        if header.startswith(sig):
            if expected_ext == "webp" and header[8:12] != b"WEBP":
                return False
            if expected_ext == "wav" and header[8:12] != b"WAVE":
                return False
            return True

    if expected_ext in ZIP_BASED and header.startswith(b"PK\x03\x04"):
        import zipfile
        try:
            with zipfile.ZipFile(file_path) as zf:
                return any(ZIP_BASED[expected_ext] in n for n in zf.namelist())
        except zipfile.BadZipFile:
            return False

    if expected_ext == "html":
        stripped = header.lstrip(b"\xef\xbb\xbf\xfe\xff ")
        if stripped.lower().startswith((b"<!doctype html", b"<html")):
            return True

    return False


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
        # Security: check for path traversal in local paths
        if not file_path.startswith(("http://", "https://", "ftp://")):
            _check_path_traversal(file_path)
        
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
        
        # Fall back to built-in extractors; verify file type if local file
        if extension in cls.BUILTIN_EXTRACTORS:
            if os.path.isfile(file_path):
                _check_zip_bomb(file_path)
                if not verify_file_type(file_path, extension):
                    log_security_event("extension_spoof", level="WARNING", file_path=file_path,
                                       reason=f"magic bytes don't match {extension}", error_code="E100")
                    raise SecurityError(f"File type mismatch: {file_path} appears to be a different format than {extension}",
                                        file_path=file_path, error_code="E100")
            module_path = cls.BUILTIN_EXTRACTORS[extension]
            return cls._import_extractor(module_path, **kwargs)
        
        # Content-based detection via magic bytes
        if os.path.isfile(file_path):
            _check_zip_bomb(file_path)
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

