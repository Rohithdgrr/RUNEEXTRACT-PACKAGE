"""Tests for WebSocket extraction server."""

import json
import os
import tempfile
import pytest

from runeextract.server import ExtractionServer


@pytest.mark.asyncio
async def test_ping():
    server = ExtractionServer()
    result = await server._process_request({"action": "ping"})
    assert result == {"status": "pong"}


@pytest.mark.asyncio
async def test_unknown_action():
    server = ExtractionServer()
    result = await server._process_request({"action": "unknown"})
    assert "error" in result


@pytest.mark.asyncio
async def test_extract_no_file():
    server = ExtractionServer()
    result = await server._process_request({"action": "extract"})
    assert "error" in result


@pytest.mark.asyncio
async def test_extract_file_not_found():
    server = ExtractionServer()
    result = await server._process_request({"action": "extract", "file_path": "/nonexistent"})
    assert "error" in result
    assert "not found" in result["error"]


@pytest.mark.asyncio
async def test_extract_file_success():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Test\nHello from server")
        path = f.name
    try:
        server = ExtractionServer()
        result = await server._process_request({"action": "extract", "file_path": path})
        assert "text" in result
        assert "Hello from server" in result["text"]
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_extract_with_options():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("test content")
        path = f.name
    try:
        server = ExtractionServer()
        result = await server._process_request({
            "action": "extract",
            "file_path": path,
            "options": {"ocr": False, "tables": False},
        })
        assert "text" in result
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_extract_bytes():
    import base64
    data = base64.b64encode(b"hello bytes").decode()
    server = ExtractionServer()
    result = await server._process_request({
        "action": "extract",
        "file_bytes": data,
        "filename": "test.md",
    })
    assert "text" in result
    assert "hello bytes" in result["text"]


@pytest.mark.asyncio
async def test_extract_bytes_too_large():
    import base64
    data = base64.b64encode(b"x" * 200).decode()
    server = ExtractionServer(max_file_size=100)
    result = await server._process_request({
        "action": "extract",
        "file_bytes": data,
        "filename": "test.pdf",
    })
    assert "error" in result
    assert "too large" in result["error"]
