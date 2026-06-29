"""WebSocket and HTTP server for real-time extraction."""

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExtractionJob:
    job_id: str
    file_path: str
    status: str = "queued"
    progress: float = 0.0
    result: Any = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class WebSocketHandler:
    def __init__(self, progress_callback: Optional[Callable] = None):
        self._progress_callback = progress_callback
        self._jobs: Dict[str, ExtractionJob] = {}

    async def handle_extract(self, websocket, message: dict):
        job_id = uuid.uuid4().hex[:12]
        file_path = message.get("file_path", "")
        options = message.get("options", {})
        if not file_path:
            await self._send(websocket, {"type": "error", "job_id": job_id, "message": "file_path required"})
            return
        job = ExtractionJob(job_id=job_id, file_path=file_path, status="queued")
        self._jobs[job_id] = job
        await self._send(websocket, {"type": "job_created", "job_id": job_id})
        asyncio.ensure_future(self._run_extraction(websocket, job, options))

    async def _run_extraction(self, websocket, job: ExtractionJob, options: dict):
        job.status = "extracting"
        job.progress = 0.05
        await self._send(websocket, {"type": "progress", "job_id": job.job_id, "progress": job.progress, "status": job.status})
        try:
            from runeextract import extract
            loop = asyncio.get_event_loop()
            job.progress = 0.3
            await self._send(websocket, {"type": "progress", "job_id": job.job_id, "progress": job.progress})
            doc = await loop.run_in_executor(None, lambda: extract(job.file_path, **options))
            job.progress = 0.9
            await self._send(websocket, {"type": "progress", "job_id": job.job_id, "progress": job.progress})
            result_data = {
                "text": doc.text,
                "source_type": doc.source_type,
                "source_path": doc.source_path,
                "metadata": doc.metadata,
                "tables": len(getattr(doc, "tables", [])),
                "images": len(getattr(doc, "images", [])),
            }
            job.result = result_data
            job.status = "completed"
            job.progress = 1.0
            await self._send(websocket, {
                "type": "result", "job_id": job.job_id,
                "status": "completed", "result": result_data,
            })
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            await self._send(websocket, {
                "type": "error", "job_id": job.job_id,
                "message": str(e),
            })

    async def _send(self, websocket, data: dict):
        if self._progress_callback:
            self._progress_callback(data)
        try:
            await websocket.send(json.dumps(data))
        except Exception:
            pass

    async def handle_message(self, websocket, raw: str):
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self._send(websocket, {"type": "error", "message": "invalid JSON"})
            return
        msg_type = msg.get("type", "")
        if msg_type == "extract":
            await self.handle_extract(websocket, msg)
        elif msg_type == "status":
            job_id = msg.get("job_id")
            job = self._jobs.get(job_id)
            if job:
                await self._send(websocket, {
                    "type": "status", "job_id": job_id,
                    "status": job.status, "progress": job.progress,
                })
            else:
                await self._send(websocket, {"type": "error", "message": f"job {job_id} not found"})
        elif msg_type == "cancel":
            job_id = msg.get("job_id")
            if job_id in self._jobs:
                self._jobs[job_id].status = "cancelled"
                await self._send(websocket, {"type": "cancelled", "job_id": job_id})
        elif msg_type == "list":
            jobs_list = [{"job_id": j.job_id, "status": j.status, "progress": j.progress} for j in self._jobs.values()]
            await self._send(websocket, {"type": "job_list", "jobs": jobs_list})
        else:
            await self._send(websocket, {"type": "error", "message": f"unknown type: {msg_type}"})


class ExtractionServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self._handler = WebSocketHandler()

    async def start(self):
        import aiohttp
        from aiohttp import web
        app = web.Application()
        app.router.add_get("/ws", self._websocket_handler)
        app.router.add_get("/health", self._health_handler)
        app.router.add_post("/extract", self._http_extract_handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        logger.info(f"Server running on ws://{self.host}:{self.port}/ws")

    async def stop(self):
        if hasattr(self, "_runner"):
            await self._runner.cleanup()

    async def _websocket_handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await self._handler.handle_message(ws, msg.data)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error(f"WS error: {ws.exception()}")
        return ws

    async def _health_handler(self, request):
        return aiohttp.web.json_response({"status": "ok"})

    async def _http_extract_handler(self, request):
        try:
            body = await request.json()
        except Exception:
            return aiohttp.web.json_response({"error": "invalid JSON"}, status=400)
        file_path = body.get("file_path", "")
        if not file_path:
            return aiohttp.web.json_response({"error": "file_path required"}, status=400)
        options = body.get("options", {})
        try:
            from runeextract import extract
            loop = asyncio.get_event_loop()
            doc = await loop.run_in_executor(None, lambda: extract(file_path, **options))
            result = {
                "text": doc.text,
                "source_type": doc.source_type,
                "source_path": doc.source_path,
                "metadata": doc.metadata,
                "tables": len(getattr(doc, "tables", [])),
                "images": len(getattr(doc, "images", [])),
            }
            return aiohttp.web.json_response({"status": "ok", "result": result})
        except Exception as e:
            return aiohttp.web.json_response({"status": "error", "message": str(e)}, status=500)


def run_server(host: str = "127.0.0.1", port: int = 8765):
    """Run the extraction server synchronously (blocking)."""
    server = ExtractionServer(host=host, port=port)
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(server.start())
        loop.run_forever()
    except KeyboardInterrupt:
        loop.run_until_complete(server.stop())
    finally:
        loop.close()


async def start_extraction_server(host: str = "127.0.0.1", port: int = 8765):
    """Start the extraction server asynchronously."""
    server = ExtractionServer(host=host, port=port)
    await server.start()
    return server
