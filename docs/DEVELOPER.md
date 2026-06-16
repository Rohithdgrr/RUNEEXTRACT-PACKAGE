# RuneExtract Developer Documentation

Developer guide for RuneExtract v0.2.0.

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
├── __init__.py              # Public API
├── config.py                # RuneExtractConfig (env/JSON/pyproject)
├── exceptions.py            # 5 custom exceptions
├── py.typed                 # PEP 561 marker
├── cli/main.py              # Argparse CLI with 14 flags
├── core/
│   ├── extractor.py         # BaseExtractor + StreamingExtractor
│   ├── router.py            # ExtractorRouter (19 builtin extensions + URL routing)
│   ├── registry.py          # ExtractorRegistry (register, discover, entry points)
│   ├── cache.py             # ExtractionCache (diskcache/JSON)
│   ├── schemas.py           # ExtractionOptions, ExtractionResult
│   └── streaming.py         # get_streaming_extractor, _WrappedStreamingExtractor
├── extractors/              # 12 extractor packages
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
│   └── notion/              # NotionExtractor (REST API)
├── processors/
│   ├── ocr.py               # OCR processor (easyocr singleton)
│   └── ai.py                # AIProcessor (OpenAI + local transformers)
├── models/
│   └── document.py          # Document, Table, Image, Chunk, ChunkingStrategy
└── tests/                   # 103 tests in 18 files
```

## Development Setup

```bash
git clone https://github.com/Rohithdgrr/RUNEEXTRACT-PACKAGE.git
cd RUNEEXTRACT-PACKAGE
pip install -e ".[dev]"
pip install -e ".[ocr,ai,youtube,notion,epub,async]"
pre-commit install
pytest                               # 103 tests
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
pytest                                    # All 103 tests
pytest -v                                 # Verbose
pytest runeextract/tests/test_models.py   # Single file
pytest -k "csv"                           # Filter by keyword
pytest --cov=runeextract                  # Coverage
```

Tests use `tempfile.NamedTemporaryFile` for temp files, `pytest.raises` for exceptions, `monkeypatch` for env vars, `pytest.mark.asyncio` for async. No mocking — tests exercise real code paths with simple inputs.

## CI/CD

- **GitHub Actions** (`.github/workflows/ci.yml`): lint (ruff + codespell), test (matrix Python 3.8–3.12), benchmark
- **pre-commit** (`.pre-commit-config.yaml`): ruff, codespell, trailing-whitespace, end-of-file-fixer
- **cibuildwheel** in `pyproject.toml` for binary wheels

## Releasing

```bash
# Update version in __init__.py and pyproject.toml
python -m build
twine check dist/*
twine upload dist/*
git tag v0.2.0
git push origin v0.2.0
```
