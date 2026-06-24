"""Tests for true async extraction (ProcessPoolExtractor, batch_process, URL extraction)."""

import pytest
from runeextract import batch_process, ProcessPoolExtractor
from runeextract.models.document import Document


def _make_doc(text: str) -> Document:
    return Document(text=text, source_type="text")


# ── batch_process ────────────────────────────────────────


@pytest.mark.asyncio
async def test_batch_process_sync_fn():
    items = [1, 2, 3]
    results = await batch_process(items, lambda x: x * 2, max_concurrency=2)
    assert results == [2, 4, 6]


@pytest.mark.asyncio
async def test_batch_process_async_fn():
    async def double(x: int) -> int:
        return x * 2

    items = [1, 2, 3]
    results = await batch_process(items, double, max_concurrency=2)
    assert results == [2, 4, 6]


@pytest.mark.asyncio
async def test_batch_process_empty():
    results = await batch_process([], lambda x: x)
    assert results == []


@pytest.mark.asyncio
async def test_batch_process_unordered():
    import asyncio

    async def slow_first(x: int) -> int:
        if x == 1:
            await asyncio.sleep(0.1)
        return x

    items = [1, 2, 3]
    results = await batch_process(items, slow_first, max_concurrency=3, preserve_order=False)
    assert set(results) == {1, 2, 3}


# ── ProcessPoolExtractor ─────────────────────────────────


@pytest.mark.asyncio
async def test_process_pool_extract():
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Pool Test\nHello from process pool.")
        path = f.name
    try:
        async with ProcessPoolExtractor(max_workers=1) as pool:
            doc = await pool.extract(path)
        assert "Pool Test" in doc.text
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_process_pool_extract_many():
    import tempfile, os
    paths = []
    for i in range(3):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
        f.write(f"# File {i}")
        f.close()
        paths.append(f.name)
    try:
        async with ProcessPoolExtractor(max_workers=2) as pool:
            docs = await pool.extract_many(paths)
        assert len(docs) == 3
    finally:
        for p in paths:
            os.unlink(p)


@pytest.mark.asyncio
async def test_process_pool_extract_nonexistent():
    async with ProcessPoolExtractor(max_workers=1) as pool:
        docs = await pool.extract_many(["/nonexistent/file.pdf"])
    assert len(docs) == 0  # errors skipped


# ── URL extraction (requires aiohttp) ────────────────────


@pytest.mark.asyncio
async def test_extract_async_url_http_error():
    """aiohttp-based download raises on HTTP error."""
    from runeextract.core.async_extractor import extract_async_url

    with pytest.raises(Exception):
        await extract_async_url("http://nonexistent.invalid/doc.pdf")


@pytest.mark.asyncio
async def test_extract_many_async_url_handles_errors():
    """Errors are caught and logged, empty list returned."""
    from runeextract.core.async_extractor import extract_many_async_url

    result = await extract_many_async_url(["http://nonexistent.invalid/doc.pdf"])
    assert isinstance(result, list)
    assert len(result) == 0  # all failed, none returned
