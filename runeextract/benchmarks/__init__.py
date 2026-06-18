"""RuneExtract Benchmarks — compare extraction speed and quality vs alternatives."""

from runeextract.benchmarks.runner import (
    BenchmarkResult, BenchmarkRunner,
    benchmark_vs_unstructured, benchmark_vs_langchain, benchmark_vs_llamaindex,
    run_all_benchmarks,
)

__all__ = [
    "BenchmarkResult", "BenchmarkRunner",
    "benchmark_vs_unstructured", "benchmark_vs_langchain", "benchmark_vs_llamaindex",
    "run_all_benchmarks",
]
