# RuneExtract API Documentation

Complete API reference for RuneExtract v0.2.0.

## Main API

### `extract()`

```python
from runeextract import extract

doc = extract(file_path, **options)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | `str` | required | File path or URL (PDF, DOCX, HTML, Markdown, CSV, JSON, Image, EPUB, YouTube, Notion) |
| `ocr` | `bool` | `False` | Enable OCR for images and scanned documents |
| `tables` | `bool` | `True` | Extract tables |
| `images` | `bool` | `True` | Extract images |
| `metadata` | `bool` | `True` | Extract metadata |
| `chunking_strategy` | `str` | `None` | `by_page`, `by_heading`, `semantic`, `fixed_size` |
| `chunk_size` | `int` | `1000` | Target chunk size in characters |
| `chunk_overlap` | `int` | `100` | Overlap between chunks |
| `progress_callback` | `callable` | `None` | `cb(stage, current, total)` |
| `**kwargs` | — | `{}` | Extractor-specific options (ocr_lang, youtube_format, etc.) |

**Returns:** `Document`

**Raises:** `ExtractionError`, `UnsupportedFormatError`, `CorruptFileError`, `FileTooLargeError`, `DependencyMissingError`

### `extract_many()`

```python
docs = extract_many(file_paths, **kwargs) -> List[Document]
```

### `extract_async()`

```python
doc = await extract_async(file_path, **kwargs) -> Document
```

### `extract_many_async()`

```python
docs = await extract_many_async(file_paths, max_concurrency=4, **kwargs) -> List[Document]
```

### `extract_stream()`

```python
async for doc in extract_stream(file_path, **kwargs) -> AsyncIterator[Document]
```

## Data Models

### Document

| Attribute | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Full text content |
| `tables` | `List[Table]` | Extracted tables |
| `images` | `List[Image]` | Extracted images |
| `metadata` | `dict` | Document metadata |
| `source_type` | `str` | File type identifier |
| `source_path` | `str` | Source file path |

**Methods:**
- `chunks(strategy, size=1000, overlap=100)` — chunk text for RAG
- `to_dict()` — serialize to dict
- `to_json()` — serialize to JSON string
- `to_markdown()` — serialize to Markdown string
- `to_langchain_documents()` — convert to LangChain Document list
- `summary(**kwargs)` — AI summary (requires AI processor)
- `keywords(**kwargs)` — AI keyword extraction
- `entities(**kwargs)` — AI entity extraction
- `questions(**kwargs)` — AI question generation
- `flashcards(**kwargs)` — AI flashcard generation

### Table

| Attribute | Type | Description |
|-----------|------|-------------|
| `rows` | `List[List[str]]` | Data rows |
| `columns` | `List[str]` | Column headers |
| `page_number` | `int` | Page number (PDF) |
| `caption` | `str` | Table caption |
| `metadata` | `dict` | Additional metadata |

**Methods:** `to_dataframe()` → `pd.DataFrame`

### Image

| Attribute | Type | Description |
|-----------|------|-------------|
| `data` | `bytes` | Raw image data |
| `format` | `str` | Image format |
| `width` | `int` | Width in pixels |
| `height` | `int` | Height in pixels |
| `page_number` | `int` | Page number (PDF) |
| `caption` | `str` | Image caption |
| `metadata` | `dict` | Additional metadata |

### Chunk

| Attribute | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Chunk text |
| `chunk_id` | `str` | Unique identifier |
| `start_index` | `int` | Start in original text |
| `end_index` | `int` | End in original text |
| `metadata` | `dict` | Strategy, size, etc. |

## Chunking Strategies

| Strategy | Value | Description |
|----------|-------|-------------|
| `BY_PAGE` | `"by_page"` | Split by page breaks |
| `BY_HEADING` | `"by_heading"` | Split by headings / `===` / `---` |
| `SEMANTIC` | `"semantic"` | Paragraph-aware chunking |
| `FIXED_SIZE` | `"fixed_size"` | Fixed-length with overlap |

## Extractors

| Extractor | Extensions | Dependencies |
|-----------|------------|--------------|
| `PDFExtractor` | `.pdf` | pymupdf, pdfplumber |
| `PdfStreamingExtractor` | `.pdf` | pymupdf, pdfplumber |
| `DocxExtractor` | `.docx`, `.doc` | python-docx |
| `PptxExtractor` | `.pptx`, `.ppt` | python-pptx |
| `XlsxExtractor` | `.xlsx`, `.xls` | openpyxl |
| `HtmlExtractor` | `.html`, `.htm` | beautifulsoup4, lxml, requests |
| `MarkdownExtractor` | `.md`, `.markdown` | markdown-it-py |
| `CsvExtractor` | `.csv` | (stdlib) |
| `JsonExtractor` | `.json` | (stdlib) |
| `ImageExtractor` | `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.bmp`, `.webp` | Pillow, easyocr (opt) |
| `EpubExtractor` | `.epub` | EbookLib, beautifulsoup4 |
| `YoutubeExtractor` | (URL-based) | youtube-transcript-api, yt-dlp |
| `NotionExtractor` | (URL-based) | requests, aiohttp (opt) |

## Exceptions

| Exception | Code | Description |
|-----------|------|-------------|
| `ExtractionError` | E000 | Base extraction error |
| `UnsupportedFormatError` | E001 | Unsupported file format |
| `CorruptFileError` | E002 | Corrupt or unreadable file |
| `FileTooLargeError` | E003 | File exceeds size limit |
| `DependencyMissingError` | E004 | Missing optional dependency |

## Configuration

```python
from runeextract.config import RuneExtractConfig, get_config, set_config

cfg = RuneExtractConfig(ocr=True, max_file_size=1000000)
set_config(cfg)

# Or via env vars:
# RUNEEXTRACT_OCR=true
# RUNEEXTRACT_MAX_FILE_SIZE=500000000
```

Resolution order (highest wins): per-call kwargs → env vars → `~/.runeextract.json` → `pyproject.toml [tool.runeextract]` → defaults.

## Cache

```python
from runeextract.core.cache import ExtractionCache

cache = ExtractionCache(cache_dir="~/.runeextract_cache", ttl=3600)
cache.set(file_path, options, document)
doc = cache.get(file_path, options)
cache.invalidate(file_path)
cache.close()
```

## Stream Processing

```python
from runeextract.core.streaming import get_streaming_extractor

extractor = get_streaming_extractor("large.pdf", ocr=False)
async for page_doc in extractor.extract_stream("large.pdf"):
    process(page_doc)
```

## CLI

```bash
runeextract [options] input [input ...]

Options:
  --ocr                  Enable OCR for images and scanned documents
  --ocr-lang LANG        OCR language(s), comma-separated (default: en)
  --no-tables            Skip table extraction
  --no-images            Skip image extraction
  --no-metadata          Skip metadata extraction
  --chunking STRAT       Chunking strategy (by_page, by_heading, semantic, fixed_size)
  --chunk-size N         Target chunk size (default: 1000)
  --chunk-overlap N      Character overlap (default: 100)
  --format, -f FMT       Output format (text, json, pretty, markdown)
  --output-dir, -o DIR   Write one output file per input
  --tree                 Show document structure tree
  --watch DIR            Watch a directory for new files
  --youtube-format FMT   YouTube format (transcript, metadata, chapters, full)
  --ai-summarize         Run AI summary after extraction
  --version, -v          Show version
```

## Registry (Plugin System)

```python
from runeextract.core.registry import ExtractorRegistry, register_extractor

@register_extractor(".xyz")
class XyzExtractor(BaseExtractor):
    def extract(self, file_path): ...
    def supported_extensions(self): return [".xyz"]

ExtractorRegistry.discover()  # Load entry-point plugins
```

## Version

```python
from runeextract import __version__
# "0.2.0"
```
