# RuneExtract API Documentation

Complete API reference for RuneExtract v0.6.0.

## Main API

### `extract()`

```python
from runeextract import extract

doc = extract(file_path, **options)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | `str` | required | File path or URL (PDF, DOCX, HTML, Markdown, CSV, JSON, Image, EPUB, YouTube, Notion, audio, video) |
| `ocr` | `bool` | `False` | Enable OCR for images and scanned documents |
| `tables` | `bool` | `True` | Extract tables |
| `images` | `bool` | `True` | Extract images |
| `metadata` | `bool` | `True` | Extract metadata |
| `chunking_strategy` | `str` | `None` | `by_page`, `by_heading`, `semantic`, `fixed_size`, `by_token`, `sentence_window` |
| `chunk_size` | `int` | `1000` | Target chunk size in characters (or tokens for `by_token`) |
| `chunk_overlap` | `int` | `100` | Overlap between chunks (chars or tokens) |
| `password` | `str` | `None` | Password for protected PDF/DOCX/XLSX files |
| `extraction_timeout` | `int` | `300` | Max seconds for extraction |
| `progress_callback` | `callable` | `None` | `cb(stage, current, total)` |
| `use_cache` | `bool` | `True` | Enable disk cache |
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
- `search(query, mode="hybrid", top_k=5, metadata_filter=None)` — search chunks
- `retrieve(query, top_k=3)` — retrieve top-k chunks
- `ask(query, **kwargs)` — RAG-enhanced question answering
- `ask_stream(query, **kwargs)` — streaming token generator
- `chat(system_prompt=None)` — create multi-turn ChatSession
- `compress(query, top_k=10)` — contextual compression
- `summary(**kwargs)` — AI summary (requires AI processor)
- `keywords(**kwargs)` — AI keyword extraction
- `entities(**kwargs)` — AI entity extraction
- `questions(**kwargs)` — AI question generation
- `flashcards(**kwargs)` — AI flashcard generation
- `redact_pii(use_dp=False, epsilon=1.0)` — PII redaction
- `score_quality()` — document quality scoring
- `to_dict()` — serialize to dict
- `to_json(indent=2)` — serialize to JSON string
- `to_markdown()` — serialize to Markdown string
- `to_openai_messages(system_message, include_images=False)` — vision-format messages
- `to_langchain_documents()` — convert to LangChain Document list
- `to_llamaindex_documents()` — convert to LlamaIndex Document list
- `to_chromadb(collection_name, persist_directory)` — index to ChromaDB
- `to_faiss(index_path)` — index to FAISS

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
| `FIXED_SIZE` | `"fixed_size"` | Fixed-length with overlap |
| `BY_PAGE` | `"by_page"` | Split by page breaks |
| `BY_HEADING` | `"by_heading"` | Split by headings / `===` / `---` |
| `SEMANTIC` | `"semantic"` | Paragraph-aware chunking |
| `BY_TOKEN` | `"by_token"` | Token-count aware (tiktoken) |
| `SENTENCE_WINDOW` | `"sentence_window"` | Group sentences into windows with overlap |

## Extractors

| Extractor | Extensions | Dependencies |
|-----------|------------|--------------|
| `PDFExtractor` | `.pdf` | pymupdf |
| `PdfStreamingExtractor` | `.pdf` | pymupdf |
| `DocxExtractor` | `.docx`, `.doc` | python-docx |
| `PptxExtractor` | `.pptx`, `.ppt` | python-pptx |
| `XlsxExtractor` | `.xlsx`, `.xls` | openpyxl |
| `HtmlExtractor` | `.html`, `.htm` | beautifulsoup4 |
| `MarkdownExtractor` | `.md`, `.markdown` | (stdlib) |
| `CsvExtractor` | `.csv` | (stdlib) |
| `JsonExtractor` | `.json` | (stdlib) |
| `ImageExtractor` | `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.bmp`, `.webp` | Pillow, easyocr (opt) |
| `EpubExtractor` | `.epub` | EbookLib, beautifulsoup4 |
| `YoutubeExtractor` | (URL-based) | youtube-transcript-api, yt-dlp |
| `NotionExtractor` | (URL-based) | requests, aiohttp (opt) |
| `AudioExtractor` | `.mp3`, `.wav`, `.flac`, `.m4a`, `.ogg`, `.wma`, `.aac`, `.opus` | openai-whisper |
| `VideoExtractor` | `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`, `.wmv` | opencv-python-headless |

## Multi-Turn Chat (ChatSession)

```python
from runeextract import extract
from runeextract.models.document import ChatSession

doc = extract("report.pdf")

# Automatic chat via Document
chat = doc.chat(system_prompt="You are a helpful document analyst.")
answer = chat.ask("What are the key findings?")

# Standalone chat (no document)
chat = ChatSession(system_prompt="You are a helpful assistant.")
chat.add_user_message("Hello!")
chat.add_assistant_message("Hi! How can I help you?")
answer = chat.ask("What is machine learning?")

# Streaming chat
for chunk in chat.ask_stream("Explain this in detail."):
    print(chunk, end="", flush=True)
```

**Methods:**
- `ask(query)` — send message with conversation history, get response
- `ask_stream(query)` — streaming version, yields tokens
- `add_user_message(text)` — manually add user message to history
- `add_assistant_message(text)` — manually add assistant response to history
- `system_prompt` — property to view/set system prompt

## Streaming AI

```python
from runeextract.processors.ai import AIProcessor

ai = AIProcessor(provider="openai", api_key="sk-...")

# Direct streaming from AIProcessor
for token in ai._call_stream("You are helpful.", "Tell me a story."):
    print(token, end="", flush=True)

# Via Document
for token in doc.ask_stream("Summarize this."):
    print(token, end="", flush=True)
```

## Exceptions

| Exception | Code | Description |
|-----------|------|-------------|
| `ExtractionError` | E000 | Base extraction error |
| `UnsupportedFormatError` | E001 | Unsupported file format |
| `CorruptFileError` | E002 | Corrupt or unreadable file |
| `FileTooLargeError` | E003 | File exceeds size limit |
| `DependencyMissingError` | E004 | Missing optional dependency |
| `ExtractionTimeoutError` | E005 | Extraction exceeded timeout |
| `PathTraversalError` | E006 | Path traversal attack detected |
| `WrongPasswordError` | E108 | Incorrect file password |
| `StructuredExtractionError` | E107 | Structured extraction failure |

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
