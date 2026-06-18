"""Tests for benchmark suite."""

import tempfile
import os

import pytest

from runeextract.benchmarks.runner import (
    BenchmarkResult, BenchmarkRunner,
    benchmark_vs_unstructured, benchmark_vs_langchain, benchmark_vs_llamaindex,
    run_all_benchmarks,
)


class TestBenchmarkResult:
    def test_create(self):
        r = BenchmarkResult(library="RuneExtract", file_path="test.pdf", duration_ms=100, text_length=500)
        assert r.library == "RuneExtract"
        assert r.chars_per_sec == 5000.0

    def test_chars_per_sec_zero(self):
        r = BenchmarkResult(library="RuneExtract", file_path="test.pdf", duration_ms=0, text_length=500)
        assert r.chars_per_sec == 0.0

    def test_chars_per_sec_negative(self):
        r = BenchmarkResult(library="RuneExtract", file_path="test.pdf", duration_ms=-1, text_length=500)
        assert r.chars_per_sec == 0.0

    def test_error_result(self):
        r = BenchmarkResult(library="RuneExtract", file_path="test.pdf", duration_ms=0, text_length=0, error="not found")
        assert r.error == "not found"


class TestBenchmarkRunner:
    def test_benchmark_extract(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("benchmark test content")
            path = f.name
        try:
            runner = BenchmarkRunner()
            result = runner.benchmark("RuneExtract", path, lambda p: "content")
            assert result.library == "RuneExtract"
            assert result.duration_ms >= 0
            assert result.text_length > 0
        finally:
            os.unlink(path)

    def test_benchmark_error(self):
        runner = BenchmarkRunner()
        def bad_fn(p):
            raise ValueError("test error")
        result = runner.benchmark("Failing", "/x.y", bad_fn)
        assert result.error == "test error"

    def test_summary(self):
        runner = BenchmarkRunner()
        runner.results.append(BenchmarkResult("A", "f1", 10, 100))
        runner.results.append(BenchmarkResult("B", "f2", 20, 200))
        s = runner.summary()
        assert "A" in s
        assert "B" in s

    def test_summary_with_error(self):
        runner = BenchmarkRunner()
        runner.results.append(BenchmarkResult("A", "f1", 10, 0, error="boom"))
        s = runner.summary()
        assert "boom" in s


class TestBenchmarkFunctions:
    def test_benchmark_vs_unstructured(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("hello")
            path = f.name
        try:
            results = benchmark_vs_unstructured(path)
            assert len(results) == 2
            assert results[0].library == "RuneExtract"
        finally:
            os.unlink(path)

    def test_run_all_benchmarks(self):
        runner = run_all_benchmarks("/nonexistent/file.pdf")
        assert len(runner.results) == 4 or len(runner.results) == 0
        # Results may be errors if competitors not installed, but should still run
        assert isinstance(runner, BenchmarkRunner)
