"""Tests for Image extractor."""

import os
import tempfile
import pytest
from runeextract import extract
from runeextract.exceptions import UnsupportedFormatError, CorruptFileError, SecurityError


def _make_png(path):
    """Create a minimal valid 1x1 red PNG using PIL."""
    from PIL import Image
    img = Image.new("RGB", (1, 1), (255, 0, 0))
    img.save(path, "PNG")


def test_image_png_metadata():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        path = tmp.name
    try:
        _make_png(path)
        doc = extract(path)
        assert doc.source_type == "image"
        assert doc.metadata.get("width") == 1
        assert doc.metadata.get("height") == 1
        assert doc.metadata.get("format") in ("PNG",)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_image_unsupported_extension():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".gif", delete=False) as f:
        f.write("not a gif")
        path = f.name
    try:
        with pytest.raises(UnsupportedFormatError):
            extract(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_image_corrupt():
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".png", delete=False) as f:
        f.write(b"not a real png file")
        path = f.name
    try:
        with pytest.raises((CorruptFileError, SecurityError)):
            extract(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)
