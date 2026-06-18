"""
RuneExtract custom exceptions with structured error codes.
"""


class ExtractionError(Exception):
    """Base exception for all extraction errors."""
    def __init__(self, message: str, file_path: str = "", error_code: str = "E000"):
        self.file_path = file_path
        self.error_code = error_code
        super().__init__(f"[{error_code}] {message}" + (f" — file: {file_path}" if file_path else ""))


class UnsupportedFormatError(ExtractionError):
    """Raised when file format is not supported."""
    def __init__(self, file_path: str, extension: str = ""):
        super().__init__(
            f"Unsupported file format: {extension or 'unknown'}",
            file_path=file_path,
            error_code="E001"
        )


class CorruptFileError(ExtractionError):
    """Raised when a file cannot be parsed due to corruption."""
    def __init__(self, file_path: str, detail: str = ""):
        msg = f"Corrupt or unreadable file" + (f": {detail}" if detail else "")
        super().__init__(msg, file_path=file_path, error_code="E002")


class FileTooLargeError(ExtractionError):
    """Raised when file exceeds configured size limit."""
    def __init__(self, file_path: str, size: int, limit: int):
        super().__init__(
            f"File too large: {size:,} bytes (limit: {limit:,} bytes)",
            file_path=file_path,
            error_code="E003"
        )


class DependencyMissingError(ExtractionError):
    """Raised when an optional dependency is not installed."""
    def __init__(self, file_path: str, dependency: str):
        super().__init__(
            f"Missing optional dependency: {dependency}. Install with: pip install runeextract[{dependency}]",
            file_path=file_path,
            error_code="E004"
        )


# --- Security error codes (E100-E199) ---

class SecurityError(ExtractionError):
    """Base class for security-related extraction errors."""
    def __init__(self, message: str, file_path: str = "", error_code: str = "E100"):
        super().__init__(message, file_path=file_path, error_code=error_code)


class URLBlockedError(SecurityError):
    """Raised when a URL is blocked by security policy."""
    def __init__(self, url: str, reason: str = ""):
        msg = f"URL blocked: {reason}" if reason else f"URL blocked: {url}"
        super().__init__(msg, file_path=url, error_code="E101")


class SSRFBlockedError(SecurityError):
    """Raised when SSRF attempt is detected."""
    def __init__(self, url: str):
        super().__init__(
            f"Access to internal/private resource blocked: {url}",
            file_path=url, error_code="E102"
        )


class PathTraversalError(SecurityError):
    """Raised when path traversal or unsafe filename is detected."""
    def __init__(self, filename: str):
        super().__init__(
            f"Unsafe filename detected: {filename}",
            file_path=filename, error_code="E103"
        )


class ExtractionTimeoutError(ExtractionError):
    """Raised when extraction exceeds the time limit."""
    def __init__(self, file_path: str, timeout_sec: int):
        super().__init__(
            f"Extraction timed out after {timeout_sec}s",
            file_path=file_path, error_code="E104"
        )


class SecretDetectedError(SecurityError):
    """Raised when a secret (API key, token, etc.) is detected in content."""
    def __init__(self, secret_type: str, context: str = ""):
        msg = f"Secret detected: {secret_type}" + (f" ({context})" if context else "")
        super().__init__(msg, file_path="", error_code="E105")


class MemoryLimitError(ExtractionError):
    """Raised when extraction exceeds the configured memory limit."""
    def __init__(self, file_path: str, used_mb: float, limit_mb: float):
        super().__init__(
            f"Memory limit exceeded: {used_mb:.1f} MB used (limit: {limit_mb:.1f} MB)",
            file_path=file_path, error_code="E106"
        )


class StructuredExtractionError(ExtractionError):
    """Raised when structured extraction (Pydantic-mapped) fails."""
    def __init__(self, message: str, file_path: str = ""):
        super().__init__(message, file_path=file_path, error_code="E107")


class WrongPasswordError(ExtractionError):
    """Raised when the wrong password is provided for a protected file."""
    def __init__(self, file_path: str):
        super().__init__(
            "Incorrect password or password required for protected file",
            file_path=file_path, error_code="E108"
        )


# --- Resource / Size Limit Errors (E300-E399) ---


class DownloadLimitError(ExtractionError):
    """Raised when a download exceeds maximum allowed size."""
    def __init__(self, file_path: str, size: int, limit: int):
        super().__init__(
            f"Download exceeded max size: {size:,} bytes (limit: {limit:,} bytes)",
            file_path=file_path, error_code="E300"
        )


class ResponseSizeError(ExtractionError):
    """Raised when a network response exceeds maximum allowed size."""
    def __init__(self, url: str, size: int, limit: int):
        super().__init__(
            f"Response too large: {size:,} bytes (limit: {limit:,} bytes)",
            file_path=url, error_code="E301"
        )


class MessageSizeError(ExtractionError):
    """Raised when a WebSocket/protocol message exceeds maximum allowed size."""
    def __init__(self, size: int, limit: int):
        super().__init__(
            f"Message too large: {size:,} bytes (limit: {limit:,} bytes)",
            error_code="E302"
        )


class ImageSizeError(ExtractionError):
    """Raised when an embedded image exceeds maximum allowed size."""
    def __init__(self, size: int, limit: int):
        super().__init__(
            f"Image data too large: {size:,} bytes (limit: {limit:,} bytes)",
            error_code="E303"
        )


# --- Circuit Breaker ---


class CircuitBreakerOpenError(ExtractionError):
    """Raised when a circuit breaker is open and the call is rejected."""
    def __init__(self, service: str):
        super().__init__(
            f"Circuit breaker open for {service} \u2014 too many failures",
            error_code="E040"
        )


# --- Bomb / Decompression Protection ---


class BombDetectionError(ExtractionError):
    """Raised when a zip bomb, decompression bomb, or similar attack is detected."""
    def __init__(self, message: str, file_path: str = "", error_code: str = "E109"):
        super().__init__(message, file_path=file_path, error_code=error_code)
