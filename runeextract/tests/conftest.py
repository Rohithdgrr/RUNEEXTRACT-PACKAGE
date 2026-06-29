import os
import tempfile
import pytest


@pytest.fixture
def md_file():
    """Create a temporary markdown file for extraction tests."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
    f.write("Hello from RuneExtract agent test!")
    f.close()
    yield f.name
    try:
        os.unlink(f.name)
    except OSError:
        pass


@pytest.fixture
def csv_file():
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    f.write("name,value\nfoo,1\nbar,2")
    f.close()
    yield f.name
    try:
        os.unlink(f.name)
    except OSError:
        pass
