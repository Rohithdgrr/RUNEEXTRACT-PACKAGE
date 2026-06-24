"""True async extraction — aiohttp-based URL downloads + ProcessPoolExecutor for CPU-bound work.

All functions are gated behind optional extras:

- ``async`` extra provides ``aiohttp`` for async HTTP downloads.
- When ``aiohttp`` is not installed, URL-based functions raise ``ImportError``.

Usage::

    from runeextract.core.async_extractor import extract_async_url, batch_process
"""

import asyncio
import logging
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, List, Optional, TypeVar

from runeextract.exceptions import DownloadLimitError

logger = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")


# Module-level helper for ProcessPoolExecutor (must be picklable)
def _extract_sync_worker(file_path: str, kwargs: dict) -> Any:
    from runeextract import extract as _sync_extract
    return _sync_extract(file_path, **kwargs)

# Default max content size for downloads (50 MB)
_DEFAULT_MAX_DOWNLOAD_SIZE = 50 * 1024 * 1024

# Shared aiohttp session singleton (created on first use)
_HTTP_SESSION: Optional["aiohttp.ClientSession"] = None


async def _get_http_session() -> "aiohttp.ClientSession":
    """Get or create a reusable aiohttp ClientSession."""
    global _HTTP_SESSION
    if _HTTP_SESSION is None or _HTTP_SESSION.closed:
        import aiohttp
        _HTTP_SESSION = aiohttp.ClientSession()
    return _HTTP_SESSION


async def _close_http_session() -> None:
    """Close the shared HTTP session if open."""
    global _HTTP_SESSION
    if _HTTP_SESSION is not None and not _HTTP_SESSION.closed:
        await _HTTP_SESSION.close()
    _HTTP_SESSION = None


# ── URL download ──────────────────────────────────────────


async def _download_url(
    url: str,
    max_size: int = _DEFAULT_MAX_DOWNLOAD_SIZE,
    session: Optional["aiohttp.ClientSession"] = None,
) -> bytes:
    """Download a URL's content as bytes using aiohttp.

    Args:
        url: URL to download.
        max_size: Maximum allowed content size.
        session: Optional pre-existing aiohttp session (created if None).

    Raises:
        ImportError: If ``aiohttp`` is not installed.
        DownloadLimitError: If response exceeds ``max_size``.
    """
    try:
        import aiohttp
    except ImportError:
        raise ImportError(
            "aiohttp is required for URL downloads. "
            "Install with: pip install runeextract[async]"
        )

    close_session = False
    if session is None:
        session = await _get_http_session()
    else:
        close_session = True

    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            resp.raise_for_status()
            cl = resp.content_length
            if cl is not None and cl > max_size:
                raise DownloadLimitError(
                    f"Download size {cl} exceeds limit {max_size}",
                    limit=max_size, actual=cl,
                )
            chunks = []
            size = 0
            async for chunk in resp.content.iter_chunked(65536):
                size += len(chunk)
                if size > max_size:
                    raise DownloadLimitError(
                        f"Download exceeded limit {max_size}",
                        limit=max_size, actual=size,
                    )
                chunks.append(chunk)
            return b"".join(chunks)
    finally:
        if close_session:
            await session.close()


async def extract_async_url(
    url: str,
    suffix: Optional[str] = None,
    max_download_size: int = _DEFAULT_MAX_DOWNLOAD_SIZE,
    session: Optional["aiohttp.ClientSession"] = None,
    **extract_kwargs: Any,
) -> Any:
    """Download a URL and extract its content asynchronously.

    Args:
        url: URL to download.
        suffix: File suffix for extraction (e.g. ``".pdf"``). If omitted,
            inferred from the URL path.
        max_download_size: Max download size in bytes (default 50 MB).
        session: Optional pre-existing aiohttp session.
        **extract_kwargs: Passed to ``runeextract.extract()``.

    Returns:
        A ``Document`` object.
    """
    import asyncio

    data = await _download_url(url, max_size=max_download_size, session=session)
    if suffix is None:
        from urllib.parse import urlparse
        suffix = os.path.splitext(urlparse(url).path)[1] or ".bin"

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            _extract_sync_worker,  # type: ignore[arg-type]
            tmp_path,
            extract_kwargs,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── Batch async URL extraction ───────────────────────────


async def extract_many_async_url(
    urls: List[str],
    max_concurrency: int = 4,
    max_download_size: int = _DEFAULT_MAX_DOWNLOAD_SIZE,
    **extract_kwargs: Any,
) -> List[Any]:
    """Download and extract multiple URLs concurrently.

    Args:
        urls: List of URLs.
        max_concurrency: Max concurrent downloads (default 4).
        max_download_size: Max per-file download size.
        **extract_kwargs: Passed to each ``extract()`` call.

    Returns:
        List of ``Document`` objects (failed extractions are skipped with a warning).
    """
    sem = asyncio.Semaphore(max_concurrency)

    async def _one(url: str) -> Optional[Any]:
        async with sem:
            try:
                return await extract_async_url(
                    url,
                    max_download_size=max_download_size,
                    **extract_kwargs,
                )
            except Exception as exc:
                logger.warning("Async URL extraction failed for %s: %s", url, exc)
                return None

    tasks = [asyncio.create_task(_one(u)) for u in urls]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


# ── ProcessPoolExecutor for CPU-bound extraction ─────────


class ProcessPoolExtractor:
    """Extract documents in a separate process pool for CPU-bound work.

    This is useful for heavy extraction tasks (PDF parsing, OCR) that
    would otherwise block the event loop even in a thread pool.

    Usage::

        from runeextract.core.async_extractor import ProcessPoolExtractor

        async with ProcessPoolExtractor(max_workers=2) as pool:
            doc = await pool.extract("large_document.pdf")
    """

    def __init__(self, max_workers: Optional[int] = None):
        self._executor = ProcessPoolExecutor(max_workers=max_workers)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def extract(self, file_path: str, **kwargs: Any) -> Any:
        """Extract a document in a subprocess.

        Returns:
            A ``Document`` object (serialized via pickle).
        """
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        return await self._loop.run_in_executor(
            self._executor, _extract_sync_worker, file_path, kwargs,
        )

    async def extract_many(
        self, file_paths: List[str], **kwargs: Any
    ) -> List[Any]:
        """Extract multiple documents, each in a subprocess."""
        tasks = [self.extract(fp, **kwargs) for fp in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: list = []
        for fp, res in zip(file_paths, results):
            if isinstance(res, Exception):
                logger.error("ProcessPool extraction failed for %s: %s", fp, res)
            else:
                out.append(res)
        return out

    async def close(self) -> None:
        """Shut down the process pool."""
        self._executor.shutdown(wait=True)

    async def __aenter__(self) -> "ProcessPoolExtractor":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


# ── Generic concurrent batch processor ───────────────────


async def batch_process(
    items: List[T],
    fn: Callable[[T], U],
    max_concurrency: int = 4,
    preserve_order: bool = True,
) -> List[U]:
    """Apply a function to each item concurrently using a semaphore.

    Args:
        items: List of input items.
        fn: Sync or async callable. If async, it will be awaited.
        max_concurrency: Max parallel executions (default 4).
        preserve_order: If True (default), results maintain item order.
            If False, results are in completion order.

    Returns:
        List of results in input order (or completion order).
    """
    sem = asyncio.Semaphore(max_concurrency)

    async def _run(item: T) -> U:
        async with sem:
            result = fn(item)
            if asyncio.iscoroutine(result):
                return await result
            return result

    tasks = [asyncio.create_task(_run(item)) for item in items]
    if preserve_order:
        return [await t for t in tasks]
    return [await t for t in asyncio.as_completed(tasks)]


# ── Cleanup helper ───────────────────────────────────────


async def cleanup() -> None:
    """Close all shared resources (HTTP session, etc.).

    Call at application shutdown::

        from runeextract.core.async_extractor import cleanup
        await cleanup()
    """
    await _close_http_session()
