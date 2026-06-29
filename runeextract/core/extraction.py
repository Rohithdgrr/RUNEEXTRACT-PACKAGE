"""
Core extraction logic moved from __init__.py for modularity.
"""

import asyncio
import hashlib
import logging
import os
import re
import stat
import tempfile
import time
import unicodedata
from collections import deque
from functools import partial
from pathlib import Path
from typing import Optional, List, Callable
from urllib.parse import urlparse, urljoin

from runeextract.core.router import ExtractorRouter, _check_path_traversal, URLValidator
from runeextract.models.document import Document, ChunkingStrategy
from runeextract.config import get_config
from runeextract.exceptions import ExtractionError, ExtractionTimeoutError
from runeextract.utils.logging import log_security_event


logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[str, int, int], None]]

_cache_instance = None

# Global memory profiler (enabled when memory_limit_mb is set in config)
_memory_profiler = None

# Shared thread pool for timeout-guarded extractions (reused across calls)
_timeout_executor = None

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

    if use_cache:
        cache = _get_cache()
        cached = cache.get(file_path, cache_safe_options)
        if cached is not None:
            logger.debug(f"Cache hit for {file_path}")
            return cached

    # Full options with password for the extractor (avoid double-dict allocation)
    cache_safe_options["password"] = password
    extractor_options = cache_safe_options

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


def _get_timeout_executor():
    global _timeout_executor
    if _timeout_executor is None:
        from concurrent.futures import ThreadPoolExecutor
        _timeout_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="extract_timeout")
    return _timeout_executor


def _run_with_timeout(func, file_path: str, timeout_sec: int):
    """Run a function with a timeout guard using a shared ThreadPoolExecutor.

    Unlike daemon threads, the executor ensures worker threads
    are properly cleaned up and don't leak after timeout.
    """
    from concurrent.futures import TimeoutError
    pool = _get_timeout_executor()
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
        if hasattr(os, "fchmod"):
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
                logger.debug(f"Failed to clean up temp file {temp_path}: {e}")
        if fd is not None:
            try:
                os.close(fd)
            except OSError as e:
                logger.debug("Failed to close temp file descriptor %s: %s", fd, e)


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
        except Exception as exc:
            logger.warning("Robots.txt fetch failed: %s", exc)
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
                except Exception as exc:
                    logger.warning("URL validation failed for %s: %s", full_url, exc)
                    continue
                if full_url not in visited:
                    to_visit.append(full_url)
        except Exception as exc:
            logger.warning("Link extraction error: %s", exc)

    return documents
