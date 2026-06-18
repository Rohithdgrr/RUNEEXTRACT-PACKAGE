"""WebSocket server for real-time document extraction.

Usage:
    python -m runeextract.server
    # Connect via WebSocket to ws://localhost:8765

Requires: websockets (optional extra)
"""

import asyncio
import json
import logging
import os
import tempfile
from typing import Optional

from runeextract.exceptions import PathTraversalError, MessageSizeError

logger = logging.getLogger(__name__)

_MAX_MESSAGE_SIZE = 100 * 1024 * 1024  # 100 MB max WebSocket message
_MAX_RESPONSE_SIZE = 50 * 1024 * 1024  # 50 MB max response


class ExtractionServer:
    """WebSocket server that accepts file paths or bytes and returns extracted content."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        max_file_size: int = 100 * 1024 * 1024,
        auth_token: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.max_file_size = max_file_size
        self.auth_token = auth_token or os.environ.get("RUNEEXTRACT_SERVER_TOKEN", "")

    def _check_auth(self, data: dict) -> bool:
        if not self.auth_token:
            return True
        return data.get("token") == self.auth_token

    async def handle(self, websocket):
        async for message in websocket:
            if len(message) > _MAX_MESSAGE_SIZE:
                await websocket.send(json.dumps({"error": f"Message too large ({len(message)} bytes)"}))
                continue
            try:
                data = json.loads(message)
                if not self._check_auth(data):
                    await websocket.send(json.dumps({"error": "Authentication failed"}))
                    logger.warning(f"Authentication failed from {websocket.remote_address}")
                    continue
                result = await self._process_request(data)
                result_json = json.dumps(result, default=str)
                if len(result_json) > _MAX_RESPONSE_SIZE:
                    await websocket.send(json.dumps({"error": "Response too large", "truncated": True}))
                else:
                    await websocket.send(result_json)
            except PathTraversalError:
                await websocket.send(json.dumps({"error": "Path traversal detected"}))
            except json.JSONDecodeError:
                await websocket.send(json.dumps({"error": "Invalid JSON"}))
            except Exception as e:
                safe_msg = str(e)[:500]
                await websocket.send(json.dumps({"error": safe_msg}))
                logger.exception("WebSocket handler error")

    async def _process_request(self, data: dict) -> dict:
        action = data.get("action", "extract")
        if action == "extract":
            return await self._handle_extract(data)
        elif action == "ping":
            return {"status": "pong"}
        else:
            return {"error": f"Unknown action: {action}"}

    async def _handle_extract(self, data: dict) -> dict:
        file_path = data.get("file_path")
        file_bytes_b64 = data.get("file_bytes")
        filename = data.get("filename", "document.pdf")
        options = data.get("options", {})

        if file_bytes_b64:
            import base64
            if len(file_bytes_b64) > _MAX_MESSAGE_SIZE:
                return {"error": "Base64 payload too large"}
            raw = base64.b64decode(file_bytes_b64)
            if len(raw) > self.max_file_size:
                return {"error": f"File too large: {len(raw)} bytes (max {self.max_file_size})"}
            suffix = os.path.splitext(filename)[1] or ".tmp"
            fd, temp_path = tempfile.mkstemp(suffix=suffix)
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(raw)
                return await self._run_extraction(temp_path, options)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        elif file_path:
            # Path traversal check
            if "\x00" in file_path:
                return {"error": "Invalid file path"}
            cleaned = file_path.replace("\\", "/")
            if "/../" in cleaned or "/.." == cleaned or cleaned.startswith("../") or cleaned == "..":
                return {"error": "Path traversal detected"}
            if not os.path.exists(file_path):
                return {"error": "File not found"}
            return await self._run_extraction(file_path, options)
        else:
            return {"error": "Provide 'file_path' or 'file_bytes'"}

    async def _run_extraction(self, file_path: str, options: dict) -> dict:
        from runeextract import extract
        loop = asyncio.get_running_loop()

        def _extract():
            doc = extract(file_path, **options)
            result = {
                "text": doc.text[:100_000] if doc.text else "",
                "tables": [[[cell for cell in row] for row in t.rows] for t in doc.tables[:100]],
                "metadata": doc.metadata,
                "source_type": doc.source_type,
                "source_path": doc.source_path,
            }
            if doc.images:
                result["image_count"] = len(doc.images)
            return result

        result = await loop.run_in_executor(None, _extract)
        return result

    async def start(self):
        try:
            import websockets
        except ImportError:
            raise ImportError("WebSocket server requires 'websockets'. Install: pip install websockets")

        logger.info(f"Starting extraction server on ws://{self.host}:{self.port}")
        if self.auth_token:
            logger.info("Authentication enabled")
        async with websockets.serve(self.handle, self.host, self.port, max_size=_MAX_MESSAGE_SIZE):
            await asyncio.Future()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    auth_token = os.environ.get("RUNEEXTRACT_SERVER_TOKEN")
    server = ExtractionServer(auth_token=auth_token)
    asyncio.run(server.start())


if __name__ == "__main__":
    main()
