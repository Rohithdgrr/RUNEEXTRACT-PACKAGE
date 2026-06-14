"""Tests for the retry decorator."""

import pytest
from unittest.mock import patch
from runeextract.utils.retry import retry


class _RetryableError(Exception):
    """Custom error for retry tests."""


def test_retry_success_first_try():
    """Function succeeds on first attempt."""
    call_count = 0

    @retry(max_attempts=3)
    def succeed():
        nonlocal call_count
        call_count += 1
        return "ok"

    result = succeed()
    assert result == "ok"
    assert call_count == 1


def test_retry_success_after_retries():
    """Function fails twice then succeeds."""
    call_count = 0

    @retry(max_attempts=3)
    def eventually_succeed():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("transient")
        return "ok"

    result = eventually_succeed()
    assert result == "ok"
    assert call_count == 3


def test_retry_exhausted():
    """Raises after all attempts fail."""
    call_count = 0

    @retry(max_attempts=3)
    def always_fails():
        nonlocal call_count
        call_count += 1
        raise ConnectionError("still failing")

    with pytest.raises(ConnectionError, match="still failing"):
        always_fails()
    assert call_count == 3


def test_retry_custom_exceptions():
    """Only catches specified exception types."""
    call_count = 0

    @retry(max_attempts=3, exceptions=(ValueError,))
    def raises_type_error():
        nonlocal call_count
        call_count += 1
        raise TypeError("not caught")

    with pytest.raises(TypeError, match="not caught"):
        raises_type_error()
    assert call_count == 1


def test_retry_custom_exceptions_caught():
    """Catches and retries on custom exception types."""
    call_count = 0

    @retry(max_attempts=3, exceptions=(_RetryableError,))
    def raises_custom():
        nonlocal call_count
        call_count += 1
        raise _RetryableError("custom transient")

    with pytest.raises(_RetryableError, match="custom transient"):
        raises_custom()
    assert call_count == 3


def test_retry_no_jitter():
    """Runs without jitter (deterministic delay)."""
    call_count = 0

    @retry(max_attempts=2, jitter=False)
    def fails():
        nonlocal call_count
        call_count += 1
        raise ConnectionError("no jitter")

    with pytest.raises(ConnectionError):
        fails()
    assert call_count == 2


def test_retry_zero_delay():
    """Zero base delay works (no sleep)."""
    call_count = 0

    @retry(max_attempts=3, base_delay=0)
    def fails():
        nonlocal call_count
        call_count += 1
        raise ConnectionError("zero delay")

    with pytest.raises(ConnectionError):
        fails()
    assert call_count == 3


def test_retry_single_attempt():
    """max_attempts=1 means no retries."""
    call_count = 0

    @retry(max_attempts=1)
    def fails():
        nonlocal call_count
        call_count += 1
        raise ConnectionError("single attempt")

    with pytest.raises(ConnectionError):
        fails()
    assert call_count == 1


def test_retry_default_exceptions():
    """Default catches ConnectionError, TimeoutError, OSError."""
    call_count = 0

    @retry()
    def raises_default():
        nonlocal call_count
        call_count += 1
        raise OSError("default catch")

    with pytest.raises(OSError):
        raises_default()
    assert call_count == 3


def test_retry_backoff_multiplier():
    """Backoff increases delay exponentially."""
    call_count = 0

    @retry(max_attempts=3, base_delay=1, backoff=4, jitter=False)
    def fails():
        nonlocal call_count
        call_count += 1
        raise ConnectionError("backoff")

    with pytest.raises(ConnectionError):
        fails()
    assert call_count == 3


def test_retry_does_not_catch_unrelated():
    """Non-matching exceptions propagate immediately."""
    call_count = 0

    @retry(max_attempts=3)
    def raises_key_error():
        nonlocal call_count
        call_count += 1
        raise KeyError("not caught")

    with pytest.raises(KeyError, match="not caught"):
        raises_key_error()
    assert call_count == 1


def test_retry_preserves_return_value():
    """Function return value is passed through on success."""
    @retry(max_attempts=3)
    def returns_value():
        return {"key": "value"}

    result = returns_value()
    assert result == {"key": "value"}
