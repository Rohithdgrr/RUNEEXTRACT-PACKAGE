"""
Structured JSON logging for security events and extraction operations.
"""

import json
import logging
import logging.handlers
import os
import traceback
from datetime import datetime, timezone
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

_SECURITY_LOGGER: Optional[logging.Logger] = None


def get_security_logger() -> logging.Logger:
    """Get or create the dedicated security event logger.

    Emits JSON-structured log records at INFO level to a rotating file
    (``~/.runeextract/security.log``) and to the root logger at WARNING.
    """
    global _SECURITY_LOGGER
    if _SECURITY_LOGGER is not None:
        return _SECURITY_LOGGER

    _SECURITY_LOGGER = logging.getLogger("runeextract.security")
    _SECURITY_LOGGER.setLevel(logging.INFO)
    _SECURITY_LOGGER.propagate = False

    log_dir = os.path.join(os.path.expanduser("~"), ".runeextract")
    os.makedirs(log_dir, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "security.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
    )
    handler.setFormatter(_JSONFormatter())
    _SECURITY_LOGGER.addHandler(handler)
    return _SECURITY_LOGGER


class _JSONFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        obj: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            obj["exc"] = "".join(traceback.format_exception(*record.exc_info)).rstrip()
        for key in ("url", "file_path", "error_code", "reason"):
            val = getattr(record, key, None)
            if val is not None:
                obj[key] = val
        return json.dumps(obj, default=str)


def log_security_event(
    event: str,
    level: str = "WARNING",
    url: Optional[str] = None,
    file_path: Optional[str] = None,
    reason: Optional[str] = None,
    error_code: Optional[str] = None,
) -> None:
    """Emit a structured security event log record.

    Args:
        event: Short human-readable event name (e.g. "ssrf_blocked", "path_traversal")
        level: Log level ("INFO", "WARNING", "ERROR")
        url: The URL involved (if applicable)
        file_path: The file path involved (if applicable)
        reason: Optional reason string
        error_code: Optional RuneExtract error code
    """
    sec_log = get_security_logger()
    extra = {k: v for k, v in {"url": url, "file_path": file_path,
                                "reason": reason, "error_code": error_code}.items()
             if v is not None}
    level_num = getattr(logging, level.upper(), logging.WARNING)
    sec_log.log(level_num, event, extra=extra)

    # Also log to root at WARNING for visibility
    if level_num >= logging.WARNING:
        parts = []
        if error_code:
            parts.append(f"[{error_code}]")
        parts.append(event)
        if url:
            parts.append(f"url={url}")
        if file_path:
            parts.append(f"path={file_path}")
        logger.warning(" ".join(parts))
