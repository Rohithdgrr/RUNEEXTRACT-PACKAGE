"""Tests for WebSocket extraction server."""

import json
from unittest.mock import AsyncMock

import pytest

from runeextract.server import ExtractionServer, WebSocketHandler


class TestExtractionServer:
    def test_default_host_port(self):
        server = ExtractionServer()
        assert server.host == "127.0.0.1"
        assert server.port == 8765

    def test_custom_host_port(self):
        server = ExtractionServer(host="0.0.0.0", port=9999)
        assert server.host == "0.0.0.0"
        assert server.port == 9999


class TestWebSocketHandler:
    @pytest.mark.asyncio
    async def test_extract_no_file_path(self):
        handler = WebSocketHandler()
        ws = AsyncMock()
        await handler.handle_extract(ws, {})
        ws.send.assert_called_once()
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "error"
        assert "file_path" in sent["message"]

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        handler = WebSocketHandler()
        ws = AsyncMock()
        await handler.handle_message(ws, "not json")
        ws.send.assert_called_once()
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "error"

    @pytest.mark.asyncio
    async def test_unknown_type(self):
        handler = WebSocketHandler()
        ws = AsyncMock()
        await handler.handle_message(ws, json.dumps({"type": "unknown_type"}))
        ws.send.assert_called_once()
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "error"
        assert "unknown" in sent["message"]

    @pytest.mark.asyncio
    async def test_status_not_found(self):
        handler = WebSocketHandler()
        ws = AsyncMock()
        await handler.handle_message(ws, json.dumps({"type": "status", "job_id": "nonexistent"}))
        ws.send.assert_called_once()
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "error"

    @pytest.mark.asyncio
    async def test_list_jobs(self):
        handler = WebSocketHandler()
        ws = AsyncMock()
        await handler.handle_message(ws, json.dumps({"type": "list"}))
        ws.send.assert_called_once()
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "job_list"

    @pytest.mark.asyncio
    async def test_cancel_job(self):
        handler = WebSocketHandler()
        ws = AsyncMock()
        await handler.handle_extract(ws, {"file_path": "/test.pdf"})
        ws.send.assert_called_once()
        sent = json.loads(ws.send.call_args[0][0])
        job_id = sent["job_id"]
        ws.reset_mock()
        await handler.handle_message(ws, json.dumps({"type": "cancel", "job_id": job_id}))
        ws.send.assert_called_once()
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "cancelled"
        assert sent["job_id"] == job_id

    @pytest.mark.asyncio
    async def test_job_created_message(self):
        handler = WebSocketHandler()
        ws = AsyncMock()
        await handler.handle_extract(ws, {"file_path": "/test.pdf"})
        ws.send.assert_called_once()
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "job_created"
        assert "job_id" in sent

    @pytest.mark.asyncio
    async def test_progress_callback(self):
        callbacks = []
        handler = WebSocketHandler(progress_callback=lambda d: callbacks.append(d))
        ws = AsyncMock()
        await handler.handle_extract(ws, {"file_path": "/test.pdf"})
        assert len(callbacks) >= 1


@pytest.mark.asyncio
async def test_start_stop_server():
    from unittest.mock import MagicMock, patch
    mock_runner = MagicMock()
    mock_runner.setup = AsyncMock()
    mock_runner.cleanup = AsyncMock()
    with patch("aiohttp.web.AppRunner", return_value=mock_runner):
        server = ExtractionServer()
        await server.start()
        await server.stop()
        assert mock_runner.cleanup.called
