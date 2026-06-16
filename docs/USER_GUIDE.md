# RuneExtract User Guide

Complete user guide for RuneExtract v0.2.0.

## Installation

```bash
pip install runeextract
pip install runeextract[ocr]       # Image/PDF OCR
pip install runeextract[ai]        # AI summarization & analysis
pip install runeextract[youtube]   # YouTube transcripts
pip install runeextract[notion]    # Notion pages
pip install runeextract[epub]      # EPUB e-books
pip install runeextract[async]     # Async HTTP
pip install runeextract[all]       # Everything
```

## Quick Start

```python
from runeextract import extract

doc = extract("document.pdf")
print(doc.text)
print(doc.tables)
print(doc.images)
print(doc.metadata)
```

## Working with Formats

### PDF

```python
doc = extract("report.pdf", ocr=True)  # OCR for scanned pages
print(doc.metadata.get("page_count"))
print(doc.metadata.get("author"))
```

### DOCX

```python
doc = extract("document.docx")
print(doc.metadata.get("title"))
print(doc.metadata.get("author"))
```

### PPTX

```python
doc = extract("slides.pptx")
print(doc.metadata.get("slide_count"))
```

### XLSX

```python
doc = extract("spreadsheet.xlsx")
print(doc.metadata.get("sheet_names"))
```

### HTML

```python
doc = extract("page.html")
doc = extract("https://example.com")
print(doc.metadata.get("title"))
```

### Markdown

```python
doc = extract("README.md")
print(doc.metadata.get("title"))  # from frontmatter
```

### CSV

```python
doc = extract("data.csv")
print(doc.metadata["row_count"])
print(doc.metadata["column_count"])
```

### JSON

```python
doc = extract("data.json")
print(doc.metadata.get("type"))   # "dict" or "list"
```

### Image

```python
doc = extract("photo.png", ocr=True)
print(doc.metadata["width"], "x", doc.metadata["height"])
print(doc.metadata["format"])
```

### EPUB

```python
doc = extract("book.epub")
print(doc.metadata.get("title"))
print(doc.metadata.get("author"))
print(doc.metadata.get("language"))
```

### YouTube

```python
doc = extract("https://youtube.com/watch?v=dQw4w9WgXcQ")
print(doc.metadata.get("title"))
print(doc.metadata.get("author"))
print(doc.text)  # transcript with timestamps
```

### Notion

```python
doc = extract("https://notion.site/my-page-abc123")
print(doc.text)
```

## Advanced Features

### Chunking for RAG

```python
chunks = doc.chunks(strategy="semantic", size=1000)
for chunk in chunks:
    print(chunk.chunk_id, chunk.text[:100])
```

Strategies: `by_page`, `by_heading` (split on `#`, `##`, `===`, `---`), `semantic` (paragraph-aware), `fixed_size` (with overlap).

### Serialization

```python
doc.to_dict()
doc.to_json()
doc.to_markdown()
doc.to_langchain_documents()
```

### AI Analysis (requires `pip install runeextract[ai]`)

```python
doc = extract("report.pdf")
print(doc.summary(max_words=200))
print(doc.keywords(top_n=10))
print(doc.entities())
print(doc.questions(n=5))
print(doc.flashcards(n=5))
```

Set `OPENAI_API_KEY` env var, or use local AI: `AIProcessor(use_local=True)`.

### OCR (requires `pip install runeextract[ocr]`)

```python
doc = extract("scanned.pdf", ocr=True)
doc = extract("photo.jpg", ocr=True, ocr_lang="en,fr")
```

### Async Processing

```python
import asyncio
from runeextract import extract_async, extract_many_async

doc = await extract_async("large.pdf")
docs = await extract_many_async(["a.pdf", "b.pdf"], max_concurrency=4)
```

### Streaming

```python
from runeextract import extract_stream

async for page_doc in extract_stream("large.pdf"):
    print(f"Page {page_doc.metadata.get('page_number')}")
```

### Progress Callback

```python
def on_progress(stage, current, total):
    print(f"{stage}: {current}/{total}")

doc = extract("report.pdf", progress_callback=on_progress)
```

### Configuration

```bash
export RUNEEXTRACT_OCR=true
export RUNEEXTRACT_MAX_FILE_SIZE=999999999
export RUNEEXTRACT_LOG_LEVEL=DEBUG
```

Or `~/.runeextract.json`:
```json
{"ocr": true, "tables": false, "images": false}
```

### Caching

```python
from runeextract.core.cache import ExtractionCache
cache = ExtractionCache(ttl=3600)
cache.set("path/to/file.pdf", {"ocr": True}, doc)
cached_doc = cache.get("path/to/file.pdf", {"ocr": True})
```

### CLI

```bash
runeextract file.pdf
runeextract file.pdf --ocr --chunking semantic --format json
runeextract file.pdf --tree
runeextract file1.pdf file2.docx file3.html -o ./output
runeextract https://youtube.com/watch?v=... --youtube-format transcript
runeextract ./watch_dir --watch
runeextract scanned.pdf --ocr --ocr-lang en,fr --ai-summarize
runeextract --version
```

## Batch Processing

```python
from runeextract import extract_many

docs = extract_many(["a.pdf", "b.docx", "c.html"])
for doc in docs:
    print(doc.source_type, len(doc.text))
```

## Error Handling

```python
from runeextract import extract
from runeextract.exceptions import (
    ExtractionError, UnsupportedFormatError,
    CorruptFileError, FileTooLargeError, DependencyMissingError
)

try:
    doc = extract("file.pdf")
except FileNotFoundError:
    print("File not found")
except UnsupportedFormatError:
    print("Format not supported")
except CorruptFileError:
    print("File is corrupted")
except FileTooLargeError:
    print("File exceeds size limit")
except DependencyMissingError as e:
    print(f"Install: pip install runeextract[{e.dependency}]")
```

## Plugin System

```python
from runeextract.core.registry import register_extractor
from runeextract.core.extractor import BaseExtractor
from runeextract.models.document import Document

@register_extractor(".txt")
class TxtExtractor(BaseExtractor):
    def extract(self, file_path: str) -> Document:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return Document(text=text, source_type="txt", source_path=file_path)

    def supported_extensions(self):
        return [".txt"]

from runeextract import extract
doc = extract("notes.txt")
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `UnsupportedFormatError` | File type not recognized; check extension |
| `CorruptFileError` | File is damaged or password-protected |
| `DependencyMissingError` | Install the required extra: `pip install runeextract[EXTRA]` |
| `FileTooLargeError` | Increase `max_file_size` in config or reduce file size |
| Empty text in PDF | Try `extract("file.pdf", ocr=True)` for scanned docs |
| `ModuleNotFoundError` | Install with the appropriate extra flag |
