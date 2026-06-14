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
