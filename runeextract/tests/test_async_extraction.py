"""Tests for async extraction."""

import os
import tempfile
import pytest


@pytest.mark.asyncio
async def test_extract_async_basic():
    from runeextract import extract_async
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write("# Async Test\nHello from async.")
        path = f.name
    try:
        doc = await extract_async(path)
        assert doc.source_type == "markdown"
        assert "Async Test" in doc.text
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_extract_many_async():
    from runeextract import extract_many_async
    paths = []
    for i in range(3):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(f"# File {i}")
            paths.append(f.name)
    try:
        docs = await extract_many_async(paths, max_concurrency=2)
        assert len(docs) == 3
        for d in docs:
            assert d.source_type == "markdown"
    finally:
        for p in paths:
            os.unlink(p)


@pytest.mark.asyncio
async def test_extract_many_async_with_errors():
    from runeextract import extract_many_async
    paths = ["nonexistent_file_12345.md"]
    docs = await extract_many_async(paths)
    assert len(docs) == 0


@pytest.mark.asyncio
async def test_extract_stream_basic():
    from runeextract import extract_stream
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write("# Stream Test\nChunk 1.\n\nChunk 2.")
        path = f.name
    try:
        count = 0
        async for partial in extract_stream(path):
            assert partial is not None
            count += 1
        assert count >= 1
    finally:
        os.unlink(path)
