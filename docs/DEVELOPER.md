# RuneExtract Developer Documentation

Developer guide for RuneExtract v0.6.0.

## Architecture

```
User API (extract, extract_many, extract_async, ...)
    ↓
ExtractorRouter (URL detection → file extension → registry → builtins)
    ↓
BaseExtractor subclass (PDF, DOCX, CSV, YouTube, ...)
    ↓
Document (unified schema: text, tables, images, metadata)
```

## Project Structure

```
runeextract/
├── __init__.py              # Public API (740 lines)
├── config.py                # RuneExtractConfig (env/JSON/pyproject)
├── exceptions.py            # Custom exception hierarchy
├── py.typed                 # PEP 561 marker
├── cli/
│   └── main.py              # Argparse CLI
├── core/
│   ├── extractor.py         # BaseExtractor + StreamingExtractor
│   ├── extraction.py        # Core extraction functions
│   ├── router.py            # ExtractorRouter (format detection + dispatch)
│   ├── registry.py          # ExtractorRegistry (register, discover, entry points)
│   ├── cache.py             # ExtractionCache (JSON + gzip)
│   ├── schemas.py           # ExtractionOptions, ExtractionResult
│   ├── streaming.py         # get_streaming_extractor
│   └── async_extractor.py   # Async extraction helpers
├── extractors/              # 14 extractor packages
│   ├── pdf/                 # PDFExtractor + PdfStreamingExtractor
│   ├── docx/                # DocxExtractor
│   ├── pptx/                # PptxExtractor
│   ├── xlsx/                # XlsxExtractor
│   ├── html/                # HtmlExtractor (file + URL)
│   ├── markdown/            # MarkdownExtractor
│   ├── csv/                 # CsvExtractor (stdlib)
│   ├── json/                # JsonExtractor (stdlib)
│   ├── image/               # ImageExtractor (Pillow + easyocr)
│   ├── epub/                # EpubExtractor (EbookLib)
│   ├── youtube/             # YoutubeExtractor (youtube-transcript-api + yt-dlp)
│   ├── notion/              # NotionExtractor (REST API)
│   ├── audio/               # AudioExtractor (Whisper)
│   └── video/               # VideoExtractor (OpenCV + Whisper)
├── processors/
│   ├── ocr.py               # OCR processor (easyocr singleton)
│   ├── ai.py                # AIProcessor (reduced: 489 lines)
│   └── providers/           # Provider modules (plugin registry)
│       ├── __init__.py      # Registry dispatch: call(), call_stream(), embed()
│       ├── openai_compat.py # OpenAI, OpenRouter, Azure, Ollama, Groq, etc.
│       ├── anthropic.py     # Anthropic Claude
│       ├── gemini.py        # Google Gemini
│       ├── bedrock.py       # AWS Bedrock
│       └── local.py         # Local transformers + sentence-transformers
├── models/
│   ├── __init__.py          # Re-exports
│   ├── document.py          # Document class (572 lines)
│   ├── types.py             # Chunk, Table, Image, ChunkingStrategy, token utils
│   ├── chunking.py          # 6 standalone chunking functions
│   └── chat_session.py      # ChatSession multi-turn conversation
├── rag/                     # RAG pipeline
│   ├── auto_pipeline.py     # Auto-RAG (zero-config)
│   ├── compressor.py        # Contextual compression
│   ├── retriever.py         # Dense/sparse/hybrid retrieval
│   ├── evaluate.py          # RAG evaluation metrics
│   └── hierarchical.py      # RAPTOR-style hierarchical chunking
├── vision/
│   └── analyzer.py          # VisionAnalyzer (describe, interpret, caption)
├── web/
│   ├── crawler.py           # SmartCrawler
│   ├── sitemap.py           # Sitemap discovery and parsing
│   └── feed.py              # RSS/Atom feed parsing
├── transform/
│   ├── pipeline.py          # DAG pipeline engine
│   └── steps.py             # 9 concrete step types
├── sync/
│   ├── watcher.py           # DirectoryWatcher
│   └── extractor.py         # FileSync, scan_and_extract, watch_and_extract
├── agent/
│   ├── mcp_server.py        # MCP server tools
│   ├── langchain.py         # RuneExtractLoader
│   ├── llamaindex.py        # RuneExtractReader
│   ├── crewai.py            # RuneExtractTool
│   └── autogen.py           # autogen_extract_tool
├── layout/
│   ├── parser.py            # LayoutParser, BoundingBox, LayoutElement
│   └── extractor.py         # parse_layout, get_reading_order
├── diff/
│   ├── analyzer.py          # DocumentComparator
│   └── extractor.py         # diff_documents, compare_files
├── embeddings/
│   └── onnx.py              # ONNXEmbeddingModel
├── storage/
│   ├── base.py              # StorageConnector ABC
│   ├── s3.py                # S3Connector
│   ├── gcs.py               # GCSConnector
│   └── azure.py             # AzureConnector
├── benchmarks/
│   └── runner.py            # BenchmarkRunner
├── dedup/
│   ├── base.py              # Deduplicator ABC
│   ├── minhash.py           # MinHashDeduplicator
│   ├── lsh.py               # LSHDeduplicator
│   └── embedding.py         # EmbeddingDeduplicator
├── server/
│   └── websocket.py         # WebSocket extraction server
├── utils/
│   ├── privacy.py           # DifferentialPrivacyEngine
│   ├── secrets.py           # scan_secrets, redact_secrets
│   ├── memory.py            # MemoryProfiler
│   ├── rate_limiter.py      # RateLimiter
│   └── maturity.py          # @beta decorator
├── toc/
│   └── __init__.py          # TOC extraction
├── citation/
│   └── analyzer.py          # CitationEngine
├── structured/
│   └── extractor.py         # StructuredExtractor, extract_structured
├── graph/
│   ├── builder.py           # GraphBuilder
│   └── extractor.py         # build_document_graph, query_graph
└── tests/                   # 733 tests across 20+ files
```

## Development Setup

```bash
git clone https://github.com/Rohithdgrr/RUNEEXTRACT-PACKAGE.git
cd RUNEEXTRACT-PACKAGE
pip install -e ".[dev]"
pip install -e ".[ocr,ai,youtube,notion,epub,async,audio,video,rag,embeddings,vector-stores]"
pre-commit install
pytest                               # 733 tests
```

## Creating an Extractor

1. Create `runeextract/extractors/<name>/extractor.py`:

```python
from runeextract.core.extractor import BaseExtractor
from runeextract.models.document import Document

class MyExtractor(BaseExtractor):
    def extract(self, file_path: str) -> Document:
        self.validate_file(file_path)
        text = open(file_path).read()
        return Document(text=text, source_type="myformat", source_path=file_path)

    def supported_extensions(self):
        return [".myext"]
```

2. Register in `runeextract/core/router.py` (BUILTIN_EXTRACTORS + ext_map)
3. Add entry point in `pyproject.toml` under `[project.entry-points."runeextract.extractors"]`
4. Add optional dependency extras if needed
5. Add tests

## Testing

```bash
pytest                                    # All 733 tests
pytest -v                                 # Verbose
pytest runeextract/tests/test_document.py # Single file
pytest -k "csv"                           # Filter by keyword
pytest --cov=runeextract --cov-report=html # Coverage report
```

Tests use `tempfile.NamedTemporaryFile` for temp files, `pytest.raises` for exceptions, `monkeypatch` for env vars, `pytest.mark.asyncio` for async. No mocking — tests exercise real code paths with simple inputs.

## CI/CD

- **GitHub Actions** (`.github/workflows/ci.yml`): lint (ruff), test (Python 3.8–3.13), coverage (Codecov), auto-release on `v*` tags
- **pre-commit** (`.pre-commit-config.yaml`): ruff (lint + format), codespell, trailing-whitespace, end-of-file-fixer

## Releasing

Releases are automated via GitHub Actions when a `v*` tag is pushed:

```bash
git tag v0.7.0
git push origin v0.7.0
```

This triggers: build → test → publish to PyPI → create GitHub Release.

### Manual release (alternative):

```bash
pip install build twine
python -m build
twine check dist/*
twine upload dist/*
git tag v0.7.0
git push origin v0.7.0
```

## Key Architecture Decisions

### Provider Plugin System
Provider-specific code is in `processors/providers/` — each module exports `create_client()`, `call()`, `call_stream()`, and optionally `embed()`. The registry dispatches via `importlib.import_module()` — no circular imports, no static import of `AIProcessor`.

### Document Refactoring
The old single-file `models/document.py` (1170 lines) was split into:
- `models/types.py` — data types (Chunk, Table, Image)
- `models/chunking.py` — 6 standalone chunking functions
- `models/chat_session.py` — ChatSession multi-turn conversation
- `models/document.py` — Document class (572 lines, imports from above)

### Lazy Loading
All optional dependencies (OCR, AI, audio, video, etc.) are loaded lazily inside functions. The package imports instantly even without any optional extras installed.
