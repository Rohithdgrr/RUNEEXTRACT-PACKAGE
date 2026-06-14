"""
Benchmarks for extraction performance.

Run with: python -m pytest runeextract/tests/benchmarks/ --benchmark-only
"""

import tempfile
import os
import pytest

pytest.importorskip("pytest_benchmark")


def _make_markdown(size_kb: int = 10) -> str:
    lines = ["# Benchmark Document\n"]
    for i in range(size_kb):
        lines.append(f"## Section {i}\n")
        lines.append(f"Content for section {i} with enough text to fill the page.\n")
        lines.append(f"| A | B | C |\n")
        lines.append(f"|---|---|---|\n")
        lines.append(f"| {i} | {i+1} | {i+2} |\n")
    return "\n".join(lines)


def test_markdown_extraction_speed(benchmark):
    content = _make_markdown(50)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write(content)
        path = f.name
    try:
        from runeextract import extract
        result = benchmark(extract, path)
        assert result.text
    finally:
        os.unlink(path)


def test_fixed_size_chunking_speed(benchmark):
    from runeextract import extract
    from runeextract.models.document import ChunkingStrategy

    content = _make_markdown(20)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write(content)
        path = f.name
    try:
        doc = extract(path)
        benchmark(doc.chunks, strategy=ChunkingStrategy.FIXED_SIZE, size=500, overlap=50)
    finally:
        os.unlink(path)
