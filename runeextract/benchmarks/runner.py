"""Benchmark runner — compare extraction speed and text output vs alternatives."""

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class BenchmarkResult:
    library: str
    file_path: str
    duration_ms: float
    text_length: int
    error: Optional[str] = None

    @property
    def chars_per_sec(self) -> float:
        if self.duration_ms <= 0:
            return 0.0
        return (self.text_length / self.duration_ms) * 1000


class BenchmarkRunner:
    """Run benchmarks comparing RuneExtract to other libraries."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def benchmark(self, label: str, file_path: str, extract_fn: Callable[[str], str]) -> BenchmarkResult:
        start = time.perf_counter()
        error = None
        text = ""
        try:
            text = extract_fn(file_path)
        except Exception as e:
            error = str(e)

        duration_ms = (time.perf_counter() - start) * 1000

        result = BenchmarkResult(
            library=label,
            file_path=file_path,
            duration_ms=duration_ms,
            text_length=len(text),
            error=error,
        )
        self.results.append(result)
        return result

    def summary(self) -> str:
        lines = ["Benchmark Summary", "=" * 60]
        for r in self.results:
            status = f"{r.library:20s} {r.file_path:30s} {r.duration_ms:8.1f}ms {r.text_length:8d}chars"
            if r.error:
                status += f" ERROR: {r.error[:50]}"
            lines.append(status)
        return "\n".join(lines)


def _runeextract_extract(file_path: str) -> str:
    from runeextract import extract
    doc = extract(file_path)
    return doc.text or ""


def _unstructured_extract(file_path: str) -> str:
    try:
        from unstructured.partition.auto import partition
        elements = partition(filename=file_path)
        return "\n".join(str(e) for e in elements)
    except ImportError:
        raise ImportError("unstructured not installed. Run: pip install unstructured")


def _langchain_extract(file_path: str) -> str:
    try:
        from langchain_community.document_loaders import UnstructuredFileLoader
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load()
        return "\n".join(d.page_content for d in docs)
    except ImportError:
        raise ImportError("langchain-community not installed. Run: pip install langchain-community")


def _llamaindex_extract(file_path: str) -> str:
    try:
        from llama_index.core import SimpleDirectoryReader
        reader = SimpleDirectoryReader(input_files=[file_path])
        docs = reader.load_data()
        return "\n".join(d.text for d in docs)
    except ImportError:
        raise ImportError("llama-index not installed. Run: pip install llama-index")


def benchmark_vs_unstructured(file_path: str) -> BenchmarkResult:
    """Benchmark RuneExtract vs Unstructured on a single file."""
    runner = BenchmarkRunner()
    runner.benchmark("RuneExtract", file_path, _runeextract_extract)
    runner.benchmark("Unstructured", file_path, _unstructured_extract)
    return runner.results


def benchmark_vs_langchain(file_path: str) -> BenchmarkResult:
    """Benchmark RuneExtract vs LangChain on a single file."""
    runner = BenchmarkRunner()
    runner.benchmark("RuneExtract", file_path, _runeextract_extract)
    runner.benchmark("LangChain", file_path, _langchain_extract)
    return runner.results


def benchmark_vs_llamaindex(file_path: str) -> BenchmarkResult:
    """Benchmark RuneExtract vs LlamaIndex on a single file."""
    runner = BenchmarkRunner()
    runner.benchmark("RuneExtract", file_path, _runeextract_extract)
    runner.benchmark("LlamaIndex", file_path, _llamaindex_extract)
    return runner.results


def run_all_benchmarks(file_path: str) -> BenchmarkRunner:
    """Run RuneExtract against all three competitors."""
    runner = BenchmarkRunner()
    runner.benchmark("RuneExtract", file_path, _runeextract_extract)
    runner.benchmark("Unstructured", file_path, _unstructured_extract)
    runner.benchmark("LangChain", file_path, _langchain_extract)
    runner.benchmark("LlamaIndex", file_path, _llamaindex_extract)
    return runner
