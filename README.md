# RuneExtract

**One extraction API for every document type.**

RuneExtract is a universal document extraction library that provides a single, consistent API for extracting content from any file type. Whether it's PDF, DOCX, HTML, images, YouTube videos, or Notion exports — RuneExtract returns the same structured output every time.

```python
from runeextract import extract

doc = extract("report.pdf")

print(doc.text)
print(doc.tables)
print(doc.images)
print(doc.metadata)
print(doc.chunks())
```

One API. Any file. Same output schema.

## Installation

```bash
pip install runeextract
```

Optional extras:

| Extra | Packages | Feature |
|-------|----------|---------|
| `[ocr]` | easyocr, Pillow | OCR for images and scanned PDFs |
| `[ai]` | openai | AI summarization, keywords, entities, Q&A, flashcards |
| `[youtube]` | youtube-transcript-api, yt-dlp | YouTube transcript and metadata |
| `[notion]` | requests, aiohttp | Notion page and database extraction |
| `[epub]` | EbookLib, beautifulsoup4 | EPUB e-book extraction |
| `[async]` | aiohttp | Async HTTP support for URL extractors |
| `[all]` | All of the above | Everything |

## Quick Start

```python
from runeextract import extract

doc = extract("report.pdf", ocr=True, tables=True, chunking_strategy="semantic")
print(doc.text)
print(doc.tables[0].to_dataframe())
```

### Batch & Async

```python
from runeextract import extract_many, extract_many_async

docs = extract_many(["a.pdf", "b.docx", "c.html"])

docs = await extract_many_async(["a.pdf", "b.docx", "c.html"], max_concurrency=4)
```

### CLI

```bash
runeextract file.pdf --ocr --chunking semantic --tree
runeextract https://youtube.com/watch?v=... --youtube-format transcript
runeextract scanned.pdf --ocr --output-dir ./output
runeextract ./docs --watch         # Watch directory for new files
```

## Supported Formats

| Format | Status | Content Extracted |
|--------|--------|-------------------|
| PDF | ✅ v0.2.0 | text, tables, images, metadata, scanned-page OCR |
| DOCX | ✅ v0.2.0 | paragraphs, tables, images, metadata, image OCR |
| PPTX | ✅ v0.2.0 | slides, tables, images, metadata, speaker notes, image OCR |
| XLSX | ✅ v0.2.0 | worksheets, tables, metadata, multiple sheets |
| HTML | ✅ v0.2.0 | headings, paragraphs, tables, links, meta tags |
| Markdown | ✅ v0.2.0 | headings, lists, code blocks, tables, frontmatter |
| CSV | ✅ v0.2.0 | tables, text, row/column metadata |
| JSON | ✅ v0.2.0 | pretty-print, table from list-of-dicts |
| Images (PNG/JPG/TIFF/BMP/WebP) | ✅ v0.2.0 | metadata (width, height, format), OCR text |
| EPUB | ✅ v0.2.0 | text, tables, images, metadata (title, author, etc.) |
| YouTube | ✅ v0.2.0 | transcript, timestamps, chapters, metadata |
| Notion | ✅ v0.2.0 | pages, databases, 14 block types, async |

## Features

### Intelligent Chunking

```python
chunks = doc.chunks(
    strategy="semantic",  # by_page, by_heading, semantic, fixed_size
    size=1000
)
```

### AI Processing (optional)

```python
doc = extract("report.pdf")
print(doc.summary())
print(doc.keywords())
print(doc.entities())
print(doc.questions())
print(doc.flashcards())
```

### OCR (optional)

```python
doc = extract("invoice.jpg", ocr=True, ocr_lang="en,fr")
```

### Streaming

```python
from runeextract import extract_stream

async for page_doc in extract_stream("large.pdf"):
    process(page_doc)
```

### Plugin System

```python
from runeextract.core.registry import register_extractor

@register_extractor(".txt")
class TxtExtractor(BaseExtractor):
    def extract(self, file_path):
        return Document(text=open(file_path).read(), source_type="txt")
```

### Configuration

```bash
export RUNEEXTRACT_OCR=true
export RUNEEXTRACT_MAX_FILE_SIZE=999999999
```

Or create `~/.runeextract.json`:
```json
{"ocr": true, "tables": false, "max_file_size": 1000000}
```

### Caching

```python
from runeextract.core.cache import ExtractionCache
cache = ExtractionCache(ttl=3600)
```

## Project Structure

```
runeextract/
├── __init__.py          # Public API (extract, extract_many, extract_async, etc.)
├── config.py            # Configuration system (env, JSON, pyproject.toml)
├── exceptions.py        # Custom exception hierarchy
├── cli/main.py          # 14 CLI flags
├── core/
│   ├── extractor.py     # BaseExtractor, StreamingExtractor
│   ├── router.py        # ExtractorRouter (19 file extensions)
│   ├── registry.py      # Plugin registry with entry-point discovery
│   ├── cache.py         # diskcache/JSON cache layer
│   ├── schemas.py       # ExtractionOptions, ExtractionResult
│   └── streaming.py     # get_streaming_extractor, wrapped fallback
├── extractors/
│   ├── pdf/             # PDF (PyMuPDF + pdfplumber)
│   ├── docx/            # DOCX (python-docx)
│   ├── pptx/            # PPTX (python-pptx)
│   ├── xlsx/            # XLSX (openpyxl)
│   ├── html/            # HTML (BeautifulSoup)
│   ├── markdown/        # Markdown (markdown-it-py)
│   ├── csv/             # CSV (stdlib csv)
│   ├── json/            # JSON (stdlib json)
│   ├── image/           # Image (Pillow + easyocr)
│   ├── epub/            # EPUB (EbookLib)
│   ├── youtube/         # YouTube (youtube-transcript-api + yt-dlp)
│   └── notion/          # Notion (REST API + aiohttp)
├── processors/
│   ├── ocr.py           # easyocr-based OCR
│   └── ai.py            # OpenAI / local transformers AI processor
├── models/
│   └── document.py      # Document, Table, Image, Chunk, ChunkingStrategy
└── tests/
    ├── 103 tests across 18 files
    └── benchmarks/
```

## Architecture

```
File/URL → Router (detects type) → Extractor → Document (unified schema)
```

## Development

```bash
pip install -e ".[dev]"
pytest                     # 103 tests
```

## License

MIT — see [LICENSE](LICENSE).

## Why RuneExtract?

Instead of learning PyMuPDF, python-docx, BeautifulSoup, EasyOCR, etc.:

```python
extract(anything)
```

That simplicity is the entire product.

---

**RuneExtract v0.2.0 — One API to extract them all.**
