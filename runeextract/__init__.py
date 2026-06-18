"""
RuneExtract - One extraction API for every document type.
"""

import hashlib
import logging
import os
import re
import stat
import tempfile
import unicodedata
from pathlib import Path
from typing import Optional, List, Callable
from runeextract.core.router import ExtractorRouter
from runeextract.models.document import Document, ChunkingStrategy
from runeextract.config import get_config
from runeextract.exceptions import ExtractionError, PathTraversalError, ExtractionTimeoutError, WrongPasswordError
from runeextract.utils.logging import log_security_event

__version__ = "0.6.0"
__all__ = [
    "extract", "extract_many", "extract_many_with_errors",
    "extract_async", "extract_many_async", "extract_and_index",
    "extract_stream", "extract_from_bytes", "extract_from_string",
    "extract_crawl",
    "Document", "ChunkingStrategy", "get_config", "set_config",
    "AutoRAG", "auto_rag",
    "scan_secrets", "redact_secrets", "MemoryProfiler",
    "DifferentialPrivacyEngine", "SecretFinding", "WrongPasswordError",
    "StructuredExtractor", "extract_structured", "StructuredExtractionError",
    "CitationEngine", "CitationResult", "cite_document",
    "SmartCrawler", "CrawlResult", "smart_crawl",
    "parse_sitemap", "discover_sitemap",
    "parse_feed", "discover_feed",
    "Pipeline", "PipelineStep", "PipelineContext", "PipelineResult", "run_pipeline",
    "DirectoryWatcher", "FileEvent", "poll_directory",
    "FileSync", "sync_directories",
    "scan_and_extract", "watch_and_extract",
    "mcp_tool_extract", "mcp_tool_extract_many", "mcp_tool_search",
    "RuneExtractLoader",
    "RuneExtractReader",
    "RuneExtractTool",
    "autogen_extract_tool",
    "LayoutElement", "BoundingBox", "LayoutParser",
    "parse_layout", "get_reading_order",
    "DiffChange", "DiffResult", "DocumentComparator",
    "diff_documents", "compare_files",
    "ONNXEmbeddingModel", "get_onnx_embedding",
    "StorageConnector", "S3Connector", "GCSConnector", "AzureConnector", "get_storage_connector",
    "MinHashDeduplicator", "LSHDeduplicator", "EmbeddingDeduplicator",
    "deduplicate", "deduplicate_documents",
    "ExtractionServer",
    "VisionAnalyzer", "ChartInterpretation", "FigureCaption",
    "describe_image", "interpret_chart", "caption_figure",
    "GraphNode", "GraphEdge", "DocumentGraph", "GraphBuilder",
    "build_document_graph", "query_graph",
    "extract_from_presigned_url",
    "TOCEntry", "TOCParser", "extract_toc", "toc_to_markdown", "toc_to_json",
    "OCRLanguageDetector", "detect_ocr_language", "get_tesseract_langs", "get_ocr_languages",
    "FastMode", "QualityLevel", "configure_quality",
]

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[str, int, int], None]]

_cache_instance = None

# Global memory profiler (enabled when memory_limit_mb is set in config)
_memory_profiler = None

# Whitelist of allowed file extensions for security validation
ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    ".html", ".htm", ".md", ".markdown", ".csv", ".json",
    ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp",
    ".epub", ".mp3", ".wav", ".flac", ".m4a", ".ogg", ".wma", ".aac", ".opus",
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv",
    ".txt", ".tmp",
}

_DEFAULT_EXTRACTION_TIMEOUT = 300  # seconds


def _get_cache():
    global _cache_instance
    if _cache_instance is None:
        from runeextract.core.cache import ExtractionCache
        _cache_instance = ExtractionCache()
    return _cache_instance


def _get_memory_profiler():
    global _memory_profiler
    if _memory_profiler is None:
        cfg = get_config()
        limit_mb = float(cfg.extra.get("memory_limit_mb", 0))
        warn_mb = float(cfg.extra.get("memory_warn_mb", 500))
        from runeextract.utils.memory import MemoryProfiler
        _memory_profiler = MemoryProfiler(warn_mb=warn_mb, limit_mb=limit_mb, enabled=limit_mb > 0)
    return _memory_profiler


def _noop_progress(stage: str, current: int, total: int) -> None:
    pass


def _sanitize_filename(filename: str, max_length: int = 255) -> str:
    """Defense-in-depth filename sanitization.

    - Unicode normalization (NFC) to prevent homograph attacks
    - Strips path components aggressively
    - Removes null bytes and control characters
    - Prevents hidden files (dot-prefix)
    - Validates extension against whitelist
    - Length caps with hash suffix
    """
    filename = unicodedata.normalize("NFC", filename)
    filename = Path(filename).name
    filename = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", filename)
    filename = re.sub(r"^\.+", "", filename)  # remove leading dots

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        log_security_event("path_traversal", level="WARNING", file_path=filename,
                           reason=f"extension '{suffix}' replaced with .tmp", error_code="E103")
        suffix = ".tmp"
        stem = re.sub(r"[^\w\-.]", "_", Path(filename).stem)
        filename = f"{stem[:max_length - 4]}{suffix}"

    if len(filename) > max_length:
        h = hashlib.sha256(filename.encode()).hexdigest()[:8]
        filename = f"{filename[:max_length - 9]}_{h}"

    return filename


def _check_path_traversal(file_path: str) -> None:
    """Detect path traversal attempts in user-supplied file paths.

    Checks for null bytes, path traversal components (..), and
    attempts to escape the current working directory.
    """
    if "\x00" in file_path:
        from runeextract.exceptions import PathTraversalError
        raise PathTraversalError(file_path)
    # Check for path traversal components
    cleaned = file_path.replace("\\", "/")
    if "/../" in cleaned or "/.." == cleaned or cleaned.startswith("../") or cleaned == "..":
        from runeextract.exceptions import PathTraversalError
        raise PathTraversalError(file_path)


def extract(
    file_path: str,
    ocr: bool = False,
    tables: bool = True,
    images: bool = True,
    metadata: bool = True,
    chunking_strategy: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    use_cache: bool = False,
    progress_callback: ProgressCallback = None,
    extraction_timeout: Optional[int] = None,
    password: Optional[str] = None,
    **kwargs
) -> Document:
    """
    Extract content from a document file.

    This is the main entry point for RuneExtract. It automatically detects
    the file type and uses the appropriate extractor.

        Args:
            file_path: Path to the file to extract (supports PDF, DOCX, HTML, Markdown, etc.)
            ocr: Enable OCR for images and scanned documents
            tables: Extract tables from the document
            images: Extract images from the document
            metadata: Extract document metadata
            chunking_strategy: Strategy for chunking text
            chunk_size: Target chunk size (characters for most strategies, tokens for by_token)
            chunk_overlap: Overlap between chunks (characters or tokens)
            use_cache: Cache the extraction result on disk (default: False)
            extraction_timeout: Maximum seconds for extraction (default: 300, 0 = no limit)
            password: Password for protected PDF, DOCX, or XLSX files
            **kwargs: Additional extractor-specific options

        Returns:
            Document object with extracted content

        Raises:
            ExtractionError: If extraction fails
    """
    cb = progress_callback or _noop_progress

    cb("resolve_config", 0, 3)
    config = get_config().merge_options(
        ocr=ocr, tables=tables, images=images, metadata=metadata,
        chunking_strategy=chunking_strategy, chunk_size=chunk_size,
        chunk_overlap=chunk_overlap, use_cache=use_cache, **kwargs
    )

    profiler = _get_memory_profiler()
    if profiler.enabled:
        mem_profile_ctx = profiler.profile(f"extract:{file_path}")
        mem_result = mem_profile_ctx.__enter__()
    else:
        mem_profile_ctx = None
        mem_result = None

    # Security: validate file path before use
    if not file_path.startswith(("http://", "https://", "ftp://")):
        _check_path_traversal(file_path)

    cb("build_options", 1, 3)
    # Build cache-safe options (never include password in cache key)
    cache_safe_options = {
            "ocr": config.ocr,
            "tables": config.tables,
            "images": config.images,
            "metadata": config.metadata,
            "chunking_strategy": config.chunking_strategy,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "max_file_size": config.max_file_size,
            **kwargs
        }
    # Full options with password for the extractor
    extractor_options = dict(cache_safe_options)
    extractor_options["password"] = password

    if use_cache:
        cache = _get_cache()
        cached = cache.get(file_path, cache_safe_options)
        if cached is not None:
            logger.debug(f"Cache hit for {file_path}")
            return cached

    cb("get_extractor", 2, 3)
    if os.path.isdir(file_path):
        raise ExtractionError(
            f"Path is a directory, not a file: {file_path}",
            file_path=file_path, error_code="E041"
        )
    extractor = ExtractorRouter.get_extractor(file_path, **extractor_options)

    cb("extract", 0, 1)
    timeout = extraction_timeout if extraction_timeout is not None else _DEFAULT_EXTRACTION_TIMEOUT
    if timeout > 0:
        document = _run_with_timeout(extractor.extract, file_path, timeout_sec=timeout)
    else:
        document = extractor.extract(file_path)

    if use_cache:
        # Cache with safe options (no password)
        cache.set(file_path, cache_safe_options, document)
        logger.debug(f"Cached result for {file_path}")

    if chunking_strategy or config.chunking_strategy:
        strategy_str = chunking_strategy or config.chunking_strategy
        strategy = ChunkingStrategy(strategy_str)
        document.chunks(strategy=strategy, size=config.chunk_size, overlap=config.chunk_overlap)

    if mem_profile_ctx is not None:
        mem_profile_ctx.__exit__(None, None, None)
        if mem_result and mem_result.exceeded_limit:
            from runeextract.exceptions import MemoryLimitError
            raise MemoryLimitError(
                file_path=file_path,
                used_mb=mem_result.after.rss_mb,
                limit_mb=float(config.extra.get("memory_limit_mb", 0)),
            )

    return document


def _run_with_timeout(func, file_path: str, timeout_sec: int):
    """Run a function with a timeout guard using ThreadPoolExecutor.

    Unlike daemon threads, ThreadPoolExecutor ensures worker threads
    are properly cleaned up and don't leak after timeout.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(func, file_path)
        try:
            return future.result(timeout=timeout_sec)
        except TimeoutError:
            log_security_event("extraction_timeout", level="ERROR", file_path=file_path,
                               reason=f"timed out after {timeout_sec}s", error_code="E104")
            future.cancel()
            raise ExtractionTimeoutError(file_path=file_path, timeout_sec=timeout_sec)


def extract_from_bytes(
    data: bytes,
    filename: str,
    ocr: bool = False,
    tables: bool = True,
    images: bool = True,
    metadata: bool = True,
    chunking_strategy: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    max_file_size: Optional[int] = None,
    **kwargs
) -> Document:
    """
    Extract content from bytes (in-memory document).

    Useful for agent workflows where content arrives as bytes from HTTP responses,
    message attachments, or database blobs.

    Args:
        data: Raw bytes of the document
        filename: Filename (with extension) for format detection
        ocr: Enable OCR for images and scanned documents
        tables: Extract tables from the document
        images: Extract images from the document
        metadata: Extract document metadata
        chunking_strategy: Strategy for chunking text
        chunk_size: Target chunk size in characters
        chunk_overlap: Character overlap between chunks
        max_file_size: Maximum allowed file size in bytes (default: 100MB)
        **kwargs: Additional extractor-specific options

    Returns:
        Document object with extracted content
    """
    size_limit = max_file_size or 100 * 1024 * 1024
    if len(data) > size_limit:
        from runeextract.exceptions import FileTooLargeError
        raise FileTooLargeError(
            file_path=filename,
            size=len(data),
            limit=size_limit,
        )

    data_bytes = data if isinstance(data, bytes) else (
        data.encode("utf-8") if isinstance(data, str) else data
    )
    safe_name = _sanitize_filename(filename)
    suffix = Path(safe_name).suffix.lower() or ".tmp"

    fd = None
    temp_path = None
    try:
        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        os.fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "wb") as f:
            f.write(data_bytes)
            f.flush()
            os.fsync(f.fileno())
        fd = None

        return extract(
            temp_path,
            ocr=ocr, tables=tables, images=images, metadata=metadata,
            chunking_strategy=chunking_strategy, chunk_size=chunk_size,
            chunk_overlap=chunk_overlap, **kwargs
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError as e:
                logger.error(f"Failed to clean up temp file {temp_path}: {e}")
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def extract_from_string(
    content: str,
    filename: str,
    ocr: bool = False,
    tables: bool = True,
    images: bool = True,
    metadata: bool = True,
    chunking_strategy: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    **kwargs
) -> Document:
    """
    Extract content from a string (in-memory document).

    Useful for agent workflows where content arrives as a string from
    web scraping, API responses, or code.

    Args:
        content: String content of the document
        filename: Filename (with extension) for format detection
        ocr: Enable OCR for images and scanned documents
        tables: Extract tables from the document
        images: Extract images from the document
        metadata: Extract document metadata
        chunking_strategy: Strategy for chunking text
        chunk_size: Target chunk size in characters
        chunk_overlap: Character overlap between chunks
        **kwargs: Additional extractor-specific options

    Returns:
        Document object with extracted content
    """
    return extract_from_bytes(
        content.encode("utf-8"), filename,
        ocr=ocr, tables=tables, images=images, metadata=metadata,
        chunking_strategy=chunking_strategy, chunk_size=chunk_size,
        chunk_overlap=chunk_overlap, **kwargs
    )


def extract_many(
    file_paths: List[str],
    progress_callback: ProgressCallback = None,
    **kwargs
) -> List[Document]:
    """
    Extract content from multiple files.

    Args:
        file_paths: List of file paths to extract
        progress_callback: Optional callback(stage, current, total)
        **kwargs: Options passed to extract()

    Returns:
        List of Document objects (failed files are skipped with warnings)
        Use extract_many_with_errors() for error details.
    """
    documents, _ = extract_many_with_errors(file_paths, progress_callback=progress_callback, **kwargs)
    return documents


_MAX_BATCH_SIZE = 1000  # prevent memory exhaustion in batch operations


def extract_many_with_errors(
    file_paths: List[str],
    progress_callback: ProgressCallback = None,
    **kwargs
):
    """
    Extract content from multiple files, returning errors alongside documents.

    Args:
        file_paths: List of file paths to extract
        progress_callback: Optional callback(stage, current, total)
        **kwargs: Options passed to extract()

    Returns:
        Tuple of (documents, errors) where errors is a list of
        dicts with 'file_path' and 'error' keys.
    """
    if len(file_paths) > _MAX_BATCH_SIZE:
        log_security_event("batch_limit", level="WARNING",
                           reason=f"batch of {len(file_paths)} exceeds limit {_MAX_BATCH_SIZE}",
                           error_code="E300")
        raise ExtractionError(
            f"Batch size {len(file_paths)} exceeds maximum of {_MAX_BATCH_SIZE}",
            error_code="E300"
        )

    documents = []
    errors = []
    total = len(file_paths)
    cb = progress_callback or _noop_progress

    for idx, file_path in enumerate(file_paths):
        try:
            cb("extract_file", idx, total)
            doc = extract(file_path, progress_callback=cb, **kwargs)
            documents.append(doc)
        except ExtractionError as e:
            logger.warning(f"Skipping {file_path}: {e}")
            errors.append({"file_path": file_path, "error": str(e), "error_code": e.error_code})
        except Exception as e:
            logger.error(f"Unexpected error extracting {file_path}", exc_info=True)
            errors.append({"file_path": file_path, "error": str(e), "error_code": "E999"})

    cb("done", total, total)
    return documents, errors


async def extract_async(
    file_path: str,
    progress_callback: ProgressCallback = None,
    **kwargs
) -> Document:
    """
    Extract content from a document file asynchronously.

    Args:
        file_path: Path to the file to extract
        progress_callback: Optional callback(stage, current, total)
        **kwargs: Options passed to extract()

    Returns:
        Document object with extracted content
    """
    import asyncio
    from functools import partial

    loop = asyncio.get_running_loop()
    fn = partial(extract, file_path, progress_callback=progress_callback, **kwargs)
    return await loop.run_in_executor(None, fn)


async def extract_many_async(
    file_paths: List[str],
    max_concurrency: int = 4,
    progress_callback: ProgressCallback = None,
    **kwargs
) -> List[Document]:
    """
    Extract content from multiple files concurrently.

    Args:
        file_paths: List of file paths to extract
        max_concurrency: Maximum concurrent extractions (default: 4)
        progress_callback: Optional callback(stage, current, total)
        **kwargs: Options passed to extract()

    Returns:
        List of Document objects
    """
    import asyncio
    sem = asyncio.Semaphore(max_concurrency)
    total = len(file_paths)
    results: List[Document] = []
    errors: List[Exception] = []
    cb = progress_callback or _noop_progress

    async def _one(idx: int, path: str):
        async with sem:
            try:
                cb("extract_file", idx, total)
                doc = await extract_async(path, **kwargs)
                results.append(doc)
            except ExtractionError as e:
                logger.warning(f"Async extraction failed for {path}: {e}")
                errors.append(e)
            except Exception as e:
                logger.error(f"Unexpected async error for {path}", exc_info=True)
                errors.append(e)

    tasks = [asyncio.create_task(_one(i, p)) for i, p in enumerate(file_paths)]
    await asyncio.gather(*tasks, return_exceptions=True)

    cb("done", total, total)
    if errors:
        logger.warning(f"{len(errors)} of {total} async extractions failed")
    return results


async def extract_stream(
    file_path: str,
    **kwargs
) -> Document:
    """
    Extract content from a document, yielding one partial Document per page/section.

    Args:
        file_path: Path to the file to extract
        **kwargs: Options passed to extract()

    Yields:
        Partial Document objects (e.g., one per page for PDFs)
    """
    from runeextract.core.streaming import get_streaming_extractor

    extractor = get_streaming_extractor(file_path, **kwargs)
    async for partial in extractor.extract_stream(file_path):
        yield partial


def extract_and_index(
    file_path: str,
    store: str = "chromadb",
    collection_name: str = "documents",
    persist_directory: str = "./chroma_db",
    chunking_strategy: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    **kwargs
) -> Document:
    """
    Extract a document, chunk it, and index it into a vector store in one call.

    Args:
        file_path: Path to the file to extract
        store: Vector store type ("chromadb" or "faiss")
        collection_name: ChromaDB collection name
        persist_directory: Directory for vector store persistence
        chunking_strategy: Chunking strategy (default: fixed_size)
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        **kwargs: Additional options passed to extract()

    Returns:
        Document object (already indexed into the vector store)
    """
    doc = extract(file_path, chunking_strategy=chunking_strategy,
                  chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
    if store == "chromadb":
        doc.to_chromadb(collection_name=collection_name, persist_directory=persist_directory)
    elif store == "faiss":
        doc.to_faiss(index_path=persist_directory.rstrip("/\\") + "/faiss_index")
    else:
        raise ValueError(f"Unknown store '{store}'. Options: chromadb, faiss")
    return doc


def extract_crawl(
    start_url: str,
    max_pages: int = 10,
    same_domain: bool = True,
    respect_robots: bool = True,
    delay: float = 0.5,
    max_response_size: int = 10 * 1024 * 1024,
    **kwargs
) -> List[Document]:
    """Crawl web pages starting from a URL and extract each as a Document.

    Discovers internal links on each page and follows them breadth-first.
    Each page is extracted via the HTML extractor.

    Args:
        start_url: Starting URL to crawl
        max_pages: Maximum number of pages to extract (default 10)
        same_domain: Only follow links to the same domain (default True)
        respect_robots: Skip URLs disallowed by robots.txt (default True)
        delay: Delay in seconds between requests (default 0.5)
        max_response_size: Max response body size in bytes (default 10MB)
        **kwargs: Additional options passed to extract()

    Returns:
        List of Document objects (one per crawled page)
    """
    import time
    from urllib.parse import urlparse, urljoin
    from collections import deque
    from runeextract.core.router import URLValidator

    visited: set = set()
    to_visit: deque = deque([start_url])
    documents: List[Document] = []
    domain = urlparse(start_url).netloc

    _rp = None
    if respect_robots:
        try:
            import urllib.robotparser
            _rp = urllib.robotparser.RobotFileParser()
            _rp.set_url(urljoin(start_url, "/robots.txt"))
            _rp.read()
        except Exception:
            pass

    def _allowed(url: str) -> bool:
        if respect_robots and _rp is not None:
            return _rp.can_fetch("*", url)
        return True

    while to_visit and len(documents) < max_pages:
        url = to_visit.popleft()
        if url in visited or not _allowed(url):
            continue
        visited.add(url)

        URLValidator.validate(url)

        try:
            doc = extract(url, **kwargs)
            documents.append(doc)
            time.sleep(delay)
        except Exception as exc:
            logger.warning(f"Crawl skipped {url}: {exc}")
            continue

        if len(documents) >= max_pages:
            break

        try:
            import requests
            from bs4 import BeautifulSoup
            resp = requests.get(url, timeout=15, allow_redirects=True)
            content_length = resp.headers.get("Content-Length")
            if content_length and int(content_length) > max_response_size:
                logger.warning(f"Skipping link discovery for {url}: response too large")
                continue
            if len(resp.content) > max_response_size:
                from runeextract.exceptions import ResponseSizeError
                raise ResponseSizeError(url, len(resp.content), max_response_size)
            soup = BeautifulSoup(resp.text, "lxml")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("#") or href.startswith("javascript:"):
                    continue
                full_url = urljoin(url, href)
                parsed = urlparse(full_url)
                if parsed.scheme not in ("http", "https"):
                    continue
                if same_domain and parsed.netloc.lower() != domain.lower():
                    continue
                # Validate all discovered URLs against SSRF
                try:
                    URLValidator.validate(full_url)
                except Exception:
                    continue
                if full_url not in visited:
                    to_visit.append(full_url)
        except Exception:
            pass

    return documents


def AutoRAG(*args, **kwargs):
    """Lazy import for :class:`runeextract.rag.AutoRAG`."""
    from runeextract.rag.auto_pipeline import AutoRAG as _AutoRAG
    return _AutoRAG(*args, **kwargs)


def auto_rag(*args, **kwargs):
    """Lazy import for :func:`runeextract.rag.auto_rag`."""
    from runeextract.rag.auto_pipeline import auto_rag as _auto_rag
    return _auto_rag(*args, **kwargs)


# --- Tier 3 Security: lazy imports ---


def scan_secrets(text: str) -> list:
    """Scan text for API keys, tokens, passwords, and other secrets."""
    from runeextract.utils.secrets import scan_secrets as _scan
    return _scan(text)


def redact_secrets(text: str, findings: list) -> str:
    """Redact detected secrets from text using finding positions."""
    from runeextract.utils.secrets import redact_secrets as _redact
    return _redact(text, findings)


def MemoryProfiler(warn_mb: float = 500.0, limit_mb: float = 0.0, enabled: bool = True):
    """Create a MemoryProfiler for profiling extraction memory usage."""
    from runeextract.utils.memory import MemoryProfiler as _MP
    return _MP(warn_mb=warn_mb, limit_mb=limit_mb, enabled=enabled)


def DifferentialPrivacyEngine(epsilon: float = 1.0, delta: float = 0.0):
    """Create a DifferentialPrivacyEngine for private PII redaction."""
    from runeextract.utils.privacy import DifferentialPrivacyEngine as _DP
    return _DP(epsilon=epsilon, delta=delta)


def SecretFinding(*args, **kwargs):
    """Lazy import for SecretFinding dataclass."""
    from runeextract.utils.secrets import SecretFinding as _SF
    return _SF(*args, **kwargs)


# --- Structured Extraction ---


def StructuredExtractor(*args, **kwargs):
    """Lazy import for :class:`runeextract.structured.StructuredExtractor`."""
    from runeextract.structured.extractor import StructuredExtractor as _SE
    return _SE(*args, **kwargs)


def extract_structured(*args, **kwargs):
    """Lazy import for :func:`runeextract.structured.extract_structured`."""
    from runeextract.structured.extractor import extract_structured as _es
    return _es(*args, **kwargs)


# --- Citation Engine ---


def CitationEngine(*args, **kwargs):
    """Lazy import for :class:`runeextract.citation.CitationEngine`."""
    from runeextract.citation.engine import CitationEngine as _CE
    return _CE(*args, **kwargs)


def CitationResult(*args, **kwargs):
    """Lazy import for :class:`runeextract.citation.CitationResult`."""
    from runeextract.citation.engine import CitationResult as _CR
    return _CR(*args, **kwargs)


def cite_document(*args, **kwargs):
    """Lazy import for :func:`runeextract.citation.cite_document`."""
    from runeextract.citation.engine import cite_document as _cd
    return _cd(*args, **kwargs)


# --- Web / Crawler ---


def SmartCrawler(*args, **kwargs):
    """Lazy import for :class:`runeextract.web.SmartCrawler`."""
    from runeextract.web.crawler import SmartCrawler as _SC
    return _SC(*args, **kwargs)


def CrawlResult(*args, **kwargs):
    """Lazy import for :class:`runeextract.web.CrawlResult`."""
    from runeextract.web.crawler import CrawlResult as _CR
    return _CR(*args, **kwargs)


def smart_crawl(*args, **kwargs):
    """Lazy import for :func:`runeextract.web.smart_crawl`."""
    from runeextract.web.crawler import smart_crawl as _sc
    return _sc(*args, **kwargs)


def parse_sitemap(*args, **kwargs):
    """Lazy import for :func:`runeextract.web.parse_sitemap`."""
    from runeextract.web.sitemap import parse_sitemap as _ps
    return _ps(*args, **kwargs)


def discover_sitemap(*args, **kwargs):
    """Lazy import for :func:`runeextract.web.discover_sitemap`."""
    from runeextract.web.sitemap import discover_sitemap as _ds
    return _ds(*args, **kwargs)


def parse_feed(*args, **kwargs):
    """Lazy import for :func:`runeextract.web.parse_feed`."""
    from runeextract.web.feed import parse_feed as _pf
    return _pf(*args, **kwargs)


def discover_feed(*args, **kwargs):
    """Lazy import for :func:`runeextract.web.discover_feed`."""
    from runeextract.web.feed import discover_feed as _df
    return _df(*args, **kwargs)


# --- Transform / Pipeline ---


def Pipeline(*args, **kwargs):
    """Lazy import for :class:`runeextract.transform.Pipeline`."""
    from runeextract.transform.pipeline import Pipeline as _P
    return _P(*args, **kwargs)


def PipelineStep(*args, **kwargs):
    """Lazy import for :class:`runeextract.transform.PipelineStep`."""
    from runeextract.transform.pipeline import PipelineStep as _PS
    return _PS(*args, **kwargs)


def PipelineContext(*args, **kwargs):
    """Lazy import for :class:`runeextract.transform.PipelineContext`."""
    from runeextract.transform.pipeline import PipelineContext as _PC
    return _PC(*args, **kwargs)


def PipelineResult(*args, **kwargs):
    """Lazy import for :class:`runeextract.transform.PipelineResult`."""
    from runeextract.transform.pipeline import PipelineResult as _PR
    return _PR(*args, **kwargs)


def run_pipeline(*args, **kwargs):
    """Lazy import for :func:`runeextract.transform.run_pipeline`."""
    from runeextract.transform.pipeline import run_pipeline as _rp
    return _rp(*args, **kwargs)


# --- File System Sync ---


def DirectoryWatcher(*args, **kwargs):
    """Lazy import for :class:`runeextract.sync.DirectoryWatcher`."""
    from runeextract.sync.watcher import DirectoryWatcher as _DW
    return _DW(*args, **kwargs)


def FileEvent(*args, **kwargs):
    """Lazy import for :class:`runeextract.sync.FileEvent`."""
    from runeextract.sync.watcher import FileEvent as _FE
    return _FE(*args, **kwargs)


def poll_directory(*args, **kwargs):
    """Lazy import for :func:`runeextract.sync.poll_directory`."""
    from runeextract.sync.watcher import poll_directory as _pd
    return _pd(*args, **kwargs)


def FileSync(*args, **kwargs):
    """Lazy import for :class:`runeextract.sync.FileSync`."""
    from runeextract.sync.syncer import FileSync as _FS
    return _FS(*args, **kwargs)


def sync_directories(*args, **kwargs):
    """Lazy import for :func:`runeextract.sync.sync_directories`."""
    from runeextract.sync.syncer import sync_directories as _sd
    return _sd(*args, **kwargs)


def scan_and_extract(*args, **kwargs):
    """Lazy import for :func:`runeextract.sync.scan_and_extract`."""
    from runeextract.sync.extractor import scan_and_extract as _se
    return _se(*args, **kwargs)


def watch_and_extract(*args, **kwargs):
    """Lazy import for :func:`runeextract.sync.watch_and_extract`."""
    from runeextract.sync.extractor import watch_and_extract as _we
    return _we(*args, **kwargs)


# --- Agent SDK Integrations ---


def mcp_tool_extract(*args, **kwargs):
    """Lazy import for :func:`runeextract.agent.mcp_tool_extract`."""
    from runeextract.agent.mcp_server import mcp_tool_extract as _mcp
    return _mcp(*args, **kwargs)


def mcp_tool_extract_many(*args, **kwargs):
    """Lazy import for :func:`runeextract.agent.mcp_tool_extract_many`."""
    from runeextract.agent.mcp_server import mcp_tool_extract_many as _mcp
    return _mcp(*args, **kwargs)


def mcp_tool_search(*args, **kwargs):
    """Lazy import for :func:`runeextract.agent.mcp_tool_search`."""
    from runeextract.agent.mcp_server import mcp_tool_search as _mcp
    return _mcp(*args, **kwargs)


def RuneExtractLoader(*args, **kwargs):
    """Lazy import for :class:`runeextract.agent.RuneExtractLoader`."""
    from runeextract.agent.langchain import RuneExtractLoader as _L
    return _L(*args, **kwargs)


def RuneExtractReader(*args, **kwargs):
    """Lazy import for :class:`runeextract.agent.RuneExtractReader`."""
    from runeextract.agent.llamaindex import RuneExtractReader as _R
    return _R(*args, **kwargs)


def RuneExtractTool(*args, **kwargs):
    """Lazy import for :class:`runeextract.agent.RuneExtractTool`."""
    from runeextract.agent.crewai import RuneExtractTool as _T
    return _T(*args, **kwargs)


def autogen_extract_tool(*args, **kwargs):
    """Lazy import for :func:`runeextract.agent.autogen_extract_tool`."""
    from runeextract.agent.autogen import autogen_extract_tool as _at
    return _at(*args, **kwargs)


# --- Layout-aware parsing ---


def LayoutElement(*args, **kwargs):
    """Lazy import for :class:`runeextract.layout.LayoutElement`."""
    from runeextract.layout.parser import LayoutElement as _LE
    return _LE(*args, **kwargs)


def BoundingBox(*args, **kwargs):
    """Lazy import for :class:`runeextract.layout.BoundingBox`."""
    from runeextract.layout.parser import BoundingBox as _BB
    return _BB(*args, **kwargs)


def LayoutParser(*args, **kwargs):
    """Lazy import for :class:`runeextract.layout.LayoutParser`."""
    from runeextract.layout.parser import LayoutParser as _LP
    return _LP(*args, **kwargs)


def parse_layout(*args, **kwargs):
    """Lazy import for :func:`runeextract.layout.parse_layout`."""
    from runeextract.layout.parser import parse_layout as _pl
    return _pl(*args, **kwargs)


def get_reading_order(*args, **kwargs):
    """Lazy import for :func:`runeextract.layout.get_reading_order`."""
    from runeextract.layout.parser import get_reading_order as _gro
    return _gro(*args, **kwargs)


# --- Document diff / version tracking ---


def DiffChange(*args, **kwargs):
    """Lazy import for :class:`runeextract.diff.DiffChange`."""
    from runeextract.diff.comparator import DiffChange as _DC
    return _DC(*args, **kwargs)


def DiffResult(*args, **kwargs):
    """Lazy import for :class:`runeextract.diff.DiffResult`."""
    from runeextract.diff.comparator import DiffResult as _DR
    return _DR(*args, **kwargs)


def DocumentComparator(*args, **kwargs):
    """Lazy import for :class:`runeextract.diff.DocumentComparator`."""
    from runeextract.diff.comparator import DocumentComparator as _DC
    return _DC(*args, **kwargs)


def diff_documents(*args, **kwargs):
    """Lazy import for :func:`runeextract.diff.diff_documents`."""
    from runeextract.diff.comparator import diff_documents as _dd
    return _dd(*args, **kwargs)


def compare_files(*args, **kwargs):
    """Lazy import for :func:`runeextract.diff.compare_files`."""
    from runeextract.diff.comparator import compare_files as _cf
    return _cf(*args, **kwargs)


# --- ONNX Embeddings ---


def ONNXEmbeddingModel(*args, **kwargs):
    """Lazy import for :class:`runeextract.embeddings.ONNXEmbeddingModel`."""
    from runeextract.embeddings.onnx import ONNXEmbeddingModel as _O
    return _O(*args, **kwargs)


def get_onnx_embedding(*args, **kwargs):
    """Lazy import for :func:`runeextract.embeddings.get_onnx_embedding`."""
    from runeextract.embeddings.onnx import get_onnx_embedding as _goe
    return _goe(*args, **kwargs)


# --- Cloud Storage Connectors ---


def StorageConnector(*args, **kwargs):
    """Lazy import for :class:`runeextract.storage.StorageConnector`."""
    from runeextract.storage.connectors import StorageConnector as _SC
    return _SC(*args, **kwargs)


def S3Connector(*args, **kwargs):
    """Lazy import for :class:`runeextract.storage.S3Connector`."""
    from runeextract.storage.connectors import S3Connector as _S3
    return _S3(*args, **kwargs)


def GCSConnector(*args, **kwargs):
    """Lazy import for :class:`runeextract.storage.GCSConnector`."""
    from runeextract.storage.connectors import GCSConnector as _GCS
    return _GCS(*args, **kwargs)


def AzureConnector(*args, **kwargs):
    """Lazy import for :class:`runeextract.storage.AzureConnector`."""
    from runeextract.storage.connectors import AzureConnector as _A
    return _A(*args, **kwargs)


def get_storage_connector(*args, **kwargs):
    """Lazy import for :func:`runeextract.storage.get_storage_connector`."""
    from runeextract.storage.connectors import get_storage_connector as _gsc
    return _gsc(*args, **kwargs)


# --- Deduplication ---


def MinHashDeduplicator(*args, **kwargs):
    """Lazy import for :class:`runeextract.dedup.MinHashDeduplicator`."""
    from runeextract.dedup.minhash import MinHashDeduplicator as _MD
    return _MD(*args, **kwargs)


def LSHDeduplicator(*args, **kwargs):
    """Lazy import for :class:`runeextract.dedup.LSHDeduplicator`."""
    from runeextract.dedup.minhash import LSHDeduplicator as _LD
    return _LD(*args, **kwargs)


def EmbeddingDeduplicator(*args, **kwargs):
    """Lazy import for :class:`runeextract.dedup.EmbeddingDeduplicator`."""
    from runeextract.dedup.minhash import EmbeddingDeduplicator as _ED
    return _ED(*args, **kwargs)


def deduplicate(*args, **kwargs):
    """Lazy import for :func:`runeextract.dedup.deduplicate`."""
    from runeextract.dedup.minhash import deduplicate as _dd
    return _dd(*args, **kwargs)


def deduplicate_documents(*args, **kwargs):
    """Lazy import for :func:`runeextract.dedup.deduplicate_documents`."""
    from runeextract.dedup.minhash import deduplicate_documents as _dd
    return _dd(*args, **kwargs)


# --- WebSocket Server ---


def ExtractionServer(*args, **kwargs):
    """Lazy import for :class:`runeextract.server.ExtractionServer`."""
    from runeextract.server import ExtractionServer as _ES
    return _ES(*args, **kwargs)


# --- Visual Document Understanding ---


def VisionAnalyzer(*args, **kwargs):
    """Lazy import for :class:`runeextract.vision.VisionAnalyzer`."""
    from runeextract.vision.analyzer import VisionAnalyzer as _VA
    return _VA(*args, **kwargs)


def ChartInterpretation(*args, **kwargs):
    """Lazy import for :class:`runeextract.vision.ChartInterpretation`."""
    from runeextract.vision.analyzer import ChartInterpretation as _CI
    return _CI(*args, **kwargs)


def FigureCaption(*args, **kwargs):
    """Lazy import for :class:`runeextract.vision.FigureCaption`."""
    from runeextract.vision.analyzer import FigureCaption as _FC
    return _FC(*args, **kwargs)


def describe_image(*args, **kwargs):
    """Lazy import for :func:`runeextract.vision.describe_image`."""
    from runeextract.vision.analyzer import describe_image as _di
    return _di(*args, **kwargs)


def interpret_chart(*args, **kwargs):
    """Lazy import for :func:`runeextract.vision.interpret_chart`."""
    from runeextract.vision.analyzer import interpret_chart as _ic
    return _ic(*args, **kwargs)


def caption_figure(*args, **kwargs):
    """Lazy import for :func:`runeextract.vision.caption_figure`."""
    from runeextract.vision.analyzer import caption_figure as _cf
    return _cf(*args, **kwargs)


# --- Document Graph / GraphRAG ---


def GraphNode(*args, **kwargs):
    """Lazy import for :class:`runeextract.graph.GraphNode`."""
    from runeextract.graph.builder import GraphNode as _GN
    return _GN(*args, **kwargs)


def GraphEdge(*args, **kwargs):
    """Lazy import for :class:`runeextract.graph.GraphEdge`."""
    from runeextract.graph.builder import GraphEdge as _GE
    return _GE(*args, **kwargs)


def DocumentGraph(*args, **kwargs):
    """Lazy import for :class:`runeextract.graph.DocumentGraph`."""
    from runeextract.graph.builder import DocumentGraph as _DG
    return _DG(*args, **kwargs)


def GraphBuilder(*args, **kwargs):
    """Lazy import for :class:`runeextract.graph.GraphBuilder`."""
    from runeextract.graph.builder import GraphBuilder as _GB
    return _GB(*args, **kwargs)


def build_document_graph(*args, **kwargs):
    """Lazy import for :func:`runeextract.graph.build_document_graph`."""
    from runeextract.graph.builder import build_document_graph as _bdg
    return _bdg(*args, **kwargs)


def query_graph(*args, **kwargs):
    """Lazy import for :func:`runeextract.graph.query_graph`."""
    from runeextract.graph.builder import query_graph as _qg
    return _qg(*args, **kwargs)


# --- Pre-signed URL Extraction ---


def extract_from_presigned_url(*args, **kwargs):
    """Lazy import for :func:`runeextract.storage.presigned.extract_from_presigned_url`."""
    from runeextract.storage.presigned import extract_from_presigned_url as _epu
    return _epu(*args, **kwargs)


# --- Table of Contents ---


def TOCEntry(*args, **kwargs):
    """Lazy import for :class:`runeextract.toc.TOCEntry`."""
    from runeextract.toc import TOCEntry as _TE
    return _TE(*args, **kwargs)


def TOCParser(*args, **kwargs):
    """Lazy import for :class:`runeextract.toc.TOCParser`."""
    from runeextract.toc import TOCParser as _TP
    return _TP(*args, **kwargs)


def extract_toc(*args, **kwargs):
    """Lazy import for :func:`runeextract.toc.extract_toc`."""
    from runeextract.toc import extract_toc as _et
    return _et(*args, **kwargs)


def toc_to_markdown(*args, **kwargs):
    """Lazy import for :func:`runeextract.toc.toc_to_markdown`."""
    from runeextract.toc import toc_to_markdown as _ttm
    return _ttm(*args, **kwargs)


def toc_to_json(*args, **kwargs):
    """Lazy import for :func:`runeextract.toc.toc_to_json`."""
    from runeextract.toc import toc_to_json as _ttj
    return _ttj(*args, **kwargs)


# --- Multi-Language OCR ---


def OCRLanguageDetector(*args, **kwargs):
    """Lazy import for :class:`runeextract.ocr.OCRLanguageDetector`."""
    from runeextract.ocr import OCRLanguageDetector as _OLD
    return _OLD(*args, **kwargs)


def detect_ocr_language(*args, **kwargs):
    """Lazy import for :func:`runeextract.ocr.detect_ocr_language`."""
    from runeextract.ocr import detect_ocr_language as _dol
    return _dol(*args, **kwargs)


def get_tesseract_langs(*args, **kwargs):
    """Lazy import for :func:`runeextract.ocr.get_tesseract_langs`."""
    from runeextract.ocr import get_tesseract_langs as _gtl
    return _gtl(*args, **kwargs)


def get_ocr_languages(*args, **kwargs):
    """Lazy import for :func:`runeextract.ocr.get_ocr_languages`."""
    from runeextract.ocr import get_ocr_languages as _gol
    return _gol(*args, **kwargs)


# --- Fast Mode / Quality Levels ---


def FastMode(*args, **kwargs):
    """Lazy import for :class:`runeextract.quality.FastMode`."""
    from runeextract.quality import FastMode as _FM
    return _FM(*args, **kwargs)


def QualityLevel(*args, **kwargs):
    """Lazy import for :class:`runeextract.quality.QualityLevel`."""
    from runeextract.quality import QualityLevel as _QL
    return _QL(*args, **kwargs)


def configure_quality(*args, **kwargs):
    """Lazy import for :func:`runeextract.quality.configure_quality`."""
    from runeextract.quality import configure_quality as _cq
    return _cq(*args, **kwargs)
