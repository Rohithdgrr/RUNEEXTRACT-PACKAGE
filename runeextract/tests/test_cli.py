"""Tests for CLI argument parsing and output."""

import os
import sys
import tempfile
import pytest
from runeextract.cli.main import build_parser


@pytest.fixture
def parser():
    return build_parser()


def test_cli_parser_basic(parser):
    args = parser.parse_args(["file.pdf"])
    assert args.input == ["file.pdf"]
    assert args.ocr is False
    assert args.format == "text"
    assert args.chunking is None


def test_cli_parser_all_flags(parser):
    args = parser.parse_args([
        "file.pdf", "--ocr", "--ocr-lang", "en,fr",
        "--no-tables", "--no-images", "--no-metadata",
        "--chunking", "semantic", "--chunk-size", "500", "--chunk-overlap", "50",
        "--format", "json", "--output-dir", "./out",
        "--tree", "--youtube-format", "transcript",
        "--ai-summarize", "--version"
    ])
    assert args.input == ["file.pdf"]
    assert args.ocr is True
    assert args.ocr_lang == "en,fr"
    assert args.tables is False
    assert args.images is False
    assert args.metadata is False
    assert args.chunking == "semantic"
    assert args.chunk_size == 500
    assert args.chunk_overlap == 50
    assert args.format == "json"
    assert args.output_dir == "./out"
    assert args.tree is True
    assert args.youtube_format == "transcript"
    assert args.ai_summarize is True
    assert args.version is True


def test_cli_parser_multiple_inputs(parser):
    args = parser.parse_args(["a.pdf", "b.docx", "c.html"])
    assert len(args.input) == 3


def test_cli_parser_empty_input(parser):
    args = parser.parse_args([])
    assert args.input == []


def test_cli_format_choices(parser):
    for fmt in ["text", "json", "pretty", "markdown"]:
        args = parser.parse_args([f"--format={fmt}", "f.pdf"])
        assert args.format == fmt


def test_cli_chunking_choices(parser):
    for strat in ["by_page", "by_heading", "semantic", "fixed_size"]:
        args = parser.parse_args([f"--chunking={strat}", "f.pdf"])
        assert args.chunking == strat


def test_cli_youtube_format_choices(parser):
    for fmt in ["transcript", "metadata", "chapters", "full"]:
        args = parser.parse_args([f"--youtube-format={fmt}", "url"])
        assert args.youtube_format == fmt


def test_cli_output_dir_creates_file(tmp_path):
    from runeextract.cli.main import main
    md_content = "# CLI Test\nHello from CLI output test."
    src = tmp_path / "test.md"
    src.write_text(md_content, encoding="utf-8")
    out_dir = tmp_path / "output"
    test_args = [
        "runeextract", str(src),
        "--output-dir", str(out_dir),
        "--format", "text"
    ]
    old_argv = sys.argv
    try:
        sys.argv = test_args
        main()
    except SystemExit:
        pass
    finally:
        sys.argv = old_argv
    assert out_dir.exists()
    files = list(out_dir.iterdir())
    assert len(files) >= 1
    content = files[0].read_text(encoding="utf-8")
    assert "CLI Test" in content
