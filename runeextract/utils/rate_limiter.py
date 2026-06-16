"""
Token-bucket rate limiter for AI API calls.

Prevents 429 errors by limiting requests per second and tokens per minute.
"""

import time
import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token-bucket rate limiter with separate request and token limits.

    Args:
        requests_per_minute: Max API requests per minute (default 60)
        tokens_per_minute: Max tokens per minute (default 0 = unlimited)
    """

    def __init__(self, requests_per_minute: int = 60, tokens_per_minute: int = 0):
        self._rmax = requests_per_minute
        self._tmax = tokens_per_minute
        self._tokens = float(requests_per_minute)
        self._token_bucket = float(tokens_per_minute) if tokens_per_minute else 0.0
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now
        self._tokens = min(self._rmax, self._tokens + elapsed * (self._rmax / 60.0))
        if self._tmax:
            self._token_bucket = min(self._tmax, self._token_bucket + elapsed * (self._tmax / 60.0))

    def acquire(self, tokens: int = 0, block: bool = True) -> bool:
        """Acquire permission to make a request.

        Args:
            tokens: Estimated token usage for this request (for token-based limiting)
            block: If True, block until capacity is available. If False, return immediately.

        Returns:
            True if acquired, False if not (only when block=False).
        """
        with self._lock:
            self._refill()
            if self._tokens < 1:
                if not block:
                    return False
                sleep_time = (1 - self._tokens) * (60.0 / self._rmax)
                time.sleep(sleep_time)
                self._refill()

            if self._tmax and self._token_bucket < tokens:
                if not block:
                    return False
                deficit = tokens - self._token_bucket
                sleep_time = deficit * (60.0 / self._tmax)
                time.sleep(sleep_time)
                self._refill()

            self._tokens -= 1.0
            if self._tmax:
                self._token_bucket -= tokens
            return True

    def __call__(self, tokens: int = 0, block: bool = True) -> bool:
        return self.acquire(tokens=tokens, block=block)
