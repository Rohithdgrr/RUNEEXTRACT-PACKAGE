"""
Exponential backoff retry decorator for transient failures.

Used by network-dependent extractors: YouTube, Notion, HTML, and OpenAI.
"""

import functools
import logging
import random
import time
from typing import Optional, Type, Tuple

logger = logging.getLogger(__name__)


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff: float = 2.0,
    jitter: bool = True,
    exceptions: Optional[Tuple[Type[Exception], ...]] = None,
):
    """
    Retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum retry attempts (default 3)
        base_delay: Initial delay in seconds (default 1.0)
        backoff: Multiplier for each retry (default 2.0)
        jitter: Add random jitter to delay (default True)
        exceptions: Tuple of exception types to catch (default ConnectionError, Timeout, OSError)
    """
    if exceptions is None:
        exceptions = (ConnectionError, TimeoutError, OSError)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_attempts - 1:
                        delay = base_delay * (backoff ** attempt)
                        if jitter:
                            delay *= 0.5 + random.random()
                        logger.debug(
                            f"Retry {attempt + 1}/{max_attempts} for {func.__name__}: {exc}. "
                            f"Waiting {delay:.2f}s"
                        )
                        time.sleep(delay)
            raise last_exc

        return wrapper

    return decorator
