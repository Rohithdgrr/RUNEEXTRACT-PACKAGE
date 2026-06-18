"""Pre-signed URL extraction — extract documents from pre-signed S3 URLs directly.

Usage:
    from runeextract import extract_from_presigned_url

    doc = extract_from_presigned_url("https://s3.amazonaws.com/bucket/key.pdf?Signature=...")
"""

import logging
import os
import tempfile
from typing import Optional

from runeextract import extract
from runeextract.core.router import URLValidator
from runeextract.exceptions import ExtractionError, URLBlockedError
from runeextract.utils.logging import log_security_event

logger = logging.getLogger(__name__)


def extract_from_presigned_url(
    url: str,
    filename: Optional[str] = None,
    max_file_size: int = 500 * 1024 * 1024,
    **kwargs,
):
    """Extract a document from a pre-signed URL.

    Downloads the file from the pre-signed URL to a temp location,
    then runs extraction.

    Args:
        url: Pre-signed S3 URL (or any direct download URL)
        filename: Optional filename hint for format detection
        max_file_size: Maximum download size in bytes (default 500MB)
        **kwargs: Options forwarded to extract()

    Returns:
        Document object

    Raises:
        ExtractionError: If download or extraction fails
    """
    URLValidator.validate(url)
    _validate_key_signing(url)

    import requests

    # Disallow redirects to prevent SSRF bypass via redirect chains
    resp = requests.get(url, stream=True, timeout=30, allow_redirects=False)
    resp.raise_for_status()

    # If a redirect occurred, validate the redirect target
    if resp.status_code in (301, 302, 303, 307, 308):
        redirect_url = resp.headers.get("Location")
        if redirect_url:
            URLValidator.validate(redirect_url)
            resp = requests.get(redirect_url, stream=True, timeout=30, allow_redirects=False)
            resp.raise_for_status()

    content_length = resp.headers.get("Content-Length")
    if content_length and int(content_length) > max_file_size:
        raise ExtractionError(
            f"File too large: {content_length} bytes (max {max_file_size})",
            file_path=url, error_code="E003",
        )

    content_type = resp.headers.get("Content-Type", "")
    if not filename:
        filename = _infer_filename(url, content_type)

    suffix = os.path.splitext(filename)[1] or ".tmp"
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    try:
        downloaded = 0
        with os.fdopen(fd, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded > max_file_size:
                    raise ExtractionError(
                        f"Download exceeded max size of {max_file_size} bytes",
                        file_path=url, error_code="E003",
                    )
        return extract(temp_path, **kwargs)
    finally:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def _validate_key_signing(url: str) -> None:
    """Basic check that the URL contains a signature/credential parameter."""
    from urllib.parse import urlparse, parse_qs
    parsed = urlparse(url)
    if parsed.scheme.startswith("s3") or "s3" in parsed.hostname or "storage" in parsed.hostname:
        qs = parse_qs(parsed.query)
        if not any(k.lower() in ("signature", "x-amz-signature", "sig", "token", "se", "st") for k in qs):
            log_security_event("unsigned_url", level="WARNING", url=url,
                               reason="no signature params in storage URL", error_code="E101")
            raise URLBlockedError(url, reason="unsigned storage URL - missing signature parameter")


def _infer_filename(url: str, content_type: str) -> str:
    import mimetypes
    from urllib.parse import urlparse, unquote

    parsed = urlparse(url)
    path = unquote(parsed.path)
    basename = os.path.basename(path)
    if basename and "." in basename:
        return basename

    ext = mimetypes.guess_extension(content_type.split(";")[0].strip()) or ".bin"
    return f"document{ext}"
