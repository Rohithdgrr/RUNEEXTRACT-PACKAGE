"""
RuneExtract - One extraction API for every document type.
"""

import logging
import os
import tempfile
from typing import Optional, List, Callable
from runeextract.core.router import ExtractorRouter
from runeextract.models.document import Document, ChunkingStrategy
from runeextract.config import get_config
from runeextract.exceptions import ExtractionError

__version__ = "0.4.0"
__all__ = [
    "extract", "extract_many", "extract_many_with_errors",
    "extract_async", "extract_many_async", "extract_and_index",
    "extract_stream", "extract_from_bytes", "extract_from_string",
    "extract_crawl",
    "Document", "ChunkingStrategy", "get_config", "set_config"
]

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[str, int, int], None]]

_cache_instance = None


def _get_cache():
    global _cache_instance
    if _cache_instance is None:
        from runeextract.core.cache import ExtractionCache
        _cache_instance = ExtractionCache()
    return _cache_instance


def _noop_progress(stage: str, current: int, total: int) -> None:
    pass


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
        chunking_strategy: Strategy for chunking text (by_page, by_heading, semantic, fixed_size, by_token)
        chunk_size: Target chunk size (characters for most strategies, tokens for by_token)
        chunk_overlap: Overlap between chunks (characters or tokens)
        use_cache: Cache the extraction result on disk (default: False)
        **kwargs: Additional extractor-specific options
        
    Returns:
        Document object with extracted content
        
    Raises:
        ExtractionError: If extraction fails
        
    Example:
        >>> from runeextract import extract
        >>> doc = extract("report.pdf")
        >>> print(doc.text)
        >>> print(doc.tables)
        >>> print(doc.chunks())
    """
    cb = progress_callback or _noop_progress

    cb("resolve_config", 0, 3)
    config = get_config().merge_options(
        ocr=ocr, tables=tables, images=images, metadata=metadata,
        chunking_strategy=chunking_strategy, chunk_size=chunk_size,
        chunk_overlap=chunk_overlap, use_cache=use_cache, **kwargs
    )

    cb("build_options", 1, 3)
    options = {
        'ocr': config.ocr,
        'tables': config.tables,
        'images': config.images,
        'metadata': config.metadata,
        'chunking_strategy': config.chunking_strategy,
        'chunk_size': config.chunk_size,
        'chunk_overlap': config.chunk_overlap,
        'max_file_size': config.max_file_size,
        **kwargs
    }

    if use_cache:
        cache = _get_cache()
        cached = cache.get(file_path, options)
        if cached is not None:
            logger.debug(f"Cache hit for {file_path}")
            return cached

    cb("get_extractor", 2, 3)
    if os.path.isdir(file_path):
        raise ExtractionError(
            f"Path is a directory, not a file: {file_path}",
            file_path=file_path, error_code="E041"
        )
    extractor = ExtractorRouter.get_extractor(file_path, **options)

    cb("extract", 0, 1)
    document = extractor.extract(file_path)

    if use_cache:
        cache.set(file_path, options, document)
        logger.debug(f"Cached result for {file_path}")

    if chunking_strategy or config.chunking_strategy:
        strategy_str = chunking_strategy or config.chunking_strategy
        strategy = ChunkingStrategy(strategy_str)
        document.chunks(strategy=strategy, size=config.chunk_size, overlap=config.chunk_overlap)

    return document


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
        **kwargs: Additional extractor-specific options
        
    Returns:
        Document object with extracted content
        
    Example:
        >>> from runeextract import extract_from_bytes
        >>> content = requests.get("https://example.com/doc.pdf").content
        >>> doc = extract_from_bytes(content, "report.pdf")
    """
    filename = os.path.basename(filename).replace('\x00', '')
    suffix = os.path.splitext(filename)[1] or ".tmp"
    if '..' in suffix or '/' in suffix or '\\' in suffix:
        suffix = ".tmp"
    write_data = data if isinstance(data, bytes) else data.encode("utf-8") if isinstance(data, str) else data
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(write_data)
        temp_path = f.name
    try:
        return extract(
            temp_path,
            ocr=ocr, tables=tables, images=images, metadata=metadata,
            chunking_strategy=chunking_strategy, chunk_size=chunk_size,
            chunk_overlap=chunk_overlap, **kwargs
        )
    finally:
        try:
            os.unlink(temp_path)
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
        
    Example:
        >>> from runeextract import extract_from_string
        >>> html = "<html><body><h1>Hello</h1></body></html>"
        >>> doc = extract_from_string(html, "page.html")
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
        **kwargs: Additional options passed to extract()

    Returns:
        List of Document objects (one per crawled page)
    """
    import time
    from urllib.parse import urlparse, urljoin
    from collections import deque

    visited: set = set()
    to_visit: deque = deque([start_url])
    documents: List[Document] = []
    domain = urlparse(start_url).netloc

    _disallowed: set = set()
    if respect_robots:
        try:
            import urllib.robotparser
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(urljoin(start_url, "/robots.txt"))
            rp.read()
            _disallowed = {urljoin(start_url, p) for p in rp.disallow_all if p}
        except Exception:
            pass

    def _allowed(url: str) -> bool:
        if respect_robots:
            for dis in _disallowed:
                if url.startswith(dis):
                    return False
        return True

    while to_visit and len(documents) < max_pages:
        url = to_visit.popleft()
        if url in visited or not _allowed(url):
            continue
        visited.add(url)

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
            resp = requests.get(url, timeout=15)
            soup = BeautifulSoup(resp.text, "lxml")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("#") or href.startswith("javascript:"):
                    continue
                full_url = urljoin(url, href)
                parsed = urlparse(full_url)
                if parsed.scheme not in ("http", "https"):
                    continue
                if same_domain and parsed.netloc != domain:
                    continue
                if full_url not in visited:
                    to_visit.append(full_url)
        except Exception:
            pass

    return documents
