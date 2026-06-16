# Usage Guide

## Installation

```bash
pip install runeextract
```

### Optional Extras

| Extra | Packages | Purpose |
|-------|----------|---------|
| `[ocr]` | easyocr, Pillow | OCR for images and scanned PDFs |
| `[ai]` | openai | AI processing (summarization, entities, etc.) |
| `[ai-anthropic]` | anthropic | Claude AI provider |
| `[ai-gemini]` | google-generativeai | Gemini AI provider |
| `[youtube]` | youtube-transcript-api, yt-dlp | YouTube extraction |
| `[notion]` | requests | Notion extraction |
| `[epub]` | EbookLib, beautifulsoup4, lxml | EPUB extraction |
| `[async]` | aiohttp | Async HTTP support |
| `[rag]` | tiktoken | Token counting for RAG |
| `[embeddings]` | sentence-transformers | Local embeddings |
| `[vector-stores]` | chromadb, faiss-cpu | Vector database integration |
| `[audio]` | openai-whisper | Audio transcription |
| `[video]` | opencv-python-headless | Video frame extraction |
| `[all]` | All of the above | Everything |

```bash
pip install "runeextract[all]"
```

## Basic Extraction

```python
from runeextract import extract

doc = extract("report.pdf")
print(doc.text)
```

### With options

```python
doc = extract(
    "scanned_invoice.pdf",
    ocr=True,                    # Enable OCR for scanned pages
    tables=True,                 # Extract tables
    images=True,                 # Extract images
    metadata=True,               # Extract document metadata
    chunking_strategy="semantic", # Chunk after extraction
    chunk_size=1000,
    use_cache=True,              # Cache results on disk
)
```

## From Bytes / String

```python
from runeextract import extract_from_bytes, extract_from_string

doc = extract_from_bytes(pdf_bytes, "document.pdf")
doc = extract_from_string("<html><body>Hello</body></html>", "page.html")
```

## Batch Extraction

```python
from runeextract import extract_many, extract_many_with_errors

docs = extract_many(["a.pdf", "b.docx", "c.html"])
docs, errors = extract_many_with_errors(["a.pdf", "b.docx", "bad.file"])
```

### Async

```python
from runeextract import extract_async, extract_many_async

doc = await extract_async("report.pdf")
docs = await extract_many_async(["a.pdf", "b.docx"], max_concurrency=4)
```

### Streaming

```python
from runeextract import extract_stream

async for page_doc in extract_stream("large.pdf"):
    print(page_doc.text)
```

## CLI

```bash
# Basic
runeextract file.pdf
runeextract file.pdf --ocr --chunking semantic

# Output
runeextract file.pdf --tree
runeextract file.pdf --json
runeextract file.pdf --markdown

# Directory output
runeextract file.pdf --output-dir ./output

# AI processing
runeextract file.pdf --ask "What is this about?"
runeextract file.pdf --summary
runeextract file.pdf --keywords
runeextract file.pdf --flashcards

# Watch mode
runeextract ./docs --watch

# YouTube
runeextract https://youtube.com/watch?v=... --youtube-format transcript

# Cache
runeextract file.pdf --no-cache
runeextract file.pdf --chunk-tokens 500
```

## Configuration

### Environment variables

```bash
export RUNEEXTRACT_OCR=true
export RUNEEXTRACT_MAX_FILE_SIZE=999999999
export RUNEEXTRACT_CHUNKING_STRATEGY=semantic
export RUNEEXTRACT_CHUNK_SIZE=2000
```

### JSON config file

Create `~/.runeextract.json`:

```json
{
    "ocr": true,
    "tables": false,
    "max_file_size": 1000000,
    "chunking_strategy": "semantic"
}
```

### Programmatic

```python
from runeextract import get_config, set_config

cfg = get_config()
cfg.ocr = True
cfg.extra["ai_model"] = "gpt-4o"
set_config(cfg)
```

## Caching

```python
from runeextract.core.cache import ExtractionCache

cache = ExtractionCache(ttl=3600)  # 1 hour TTL
cache.invalidate("report.pdf")     # Clear cached entry
```

## Plugin System

```python
from runeextract.core.registry import register_extractor
from runeextract.core.extractor import BaseExtractor
from runeextract.models.document import Document

@register_extractor(".txt")
class TxtExtractor(BaseExtractor):
    def extract(self, file_path):
        with open(file_path) as f:
            return Document(text=f.read(), source_type="txt")

    def supported_extensions(self):
        return [".txt"]
```

## Error Handling

```python
from runeextract.exceptions import (
    ExtractionError, UnsupportedFormatError,
    CorruptFileError, FileTooLargeError, DependencyMissingError
)

try:
    doc = extract("bad_file.xyz")
except UnsupportedFormatError:
    print("Format not supported")
except CorruptFileError:
    print("File is corrupt")
except FileTooLargeError:
    print("File exceeds size limit")
except DependencyMissingError as e:
    print(f"Missing dependency: {e}")
except ExtractionError as e:
    print(f"Extraction failed: {e} (code: {e.error_code})")
```
