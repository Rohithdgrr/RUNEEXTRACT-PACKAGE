# RuneExtract

**One extraction API for every document type.** v0.6.0-dev

RuneExtract is a universal document extraction library that provides a single, consistent API for extracting content from any file type — PDFs, DOCX, HTML, images, audio, video, YouTube, Notion, and more.

```python
from runeextract import extract

doc = extract("report.pdf")
print(doc.text)
print(doc.tables)
print(doc.chunks())
```

One API. Any file. Same output schema.

## Quick Install

```bash
pip install runeextract
pip install "runeextract[all]"   # Everything (OCR, AI, audio, video, RAG)
```

## Quick Start

```python
from runeextract import extract

# Extract any file — same API
doc = extract("report.pdf")
doc = extract("presentation.pptx", images=True)
doc = extract("https://youtube.com/watch?v=...")
doc = extract("audio.mp3")
doc = extract("video.mp4")

# Access content
print(doc.text[:500])     # Plain text
print(doc.tables[0].rows) # Tables
print(doc.metadata)        # Title, author, dates, etc.

# RAG
doc.chunks(strategy="sentence_window", size=5)
results = doc.search("machine learning", mode="hybrid")
answer = doc.ask("What are the key findings?")

# AI
print(doc.summary())       # AI summary
print(doc.entities())      # Named entities

# Export
doc.to_chromadb(collection_name="docs")
print(doc.to_json(indent=2))
```

## Supported Formats

| Format | Extensions | Extracted Content |
|--------|-----------|-------------------|
| PDF | .pdf | text, tables, images, metadata, OCR |
| DOCX | .docx, .doc | paragraphs, tables, images, metadata |
| PPTX | .pptx, .ppt | slides, tables, images, speaker notes |
| XLSX | .xlsx, .xls | worksheets, tables, multiple sheets |
| HTML | .html, .htm | headings, tables, links, meta tags |
| Markdown | .md, .markdown | headings, code blocks, tables, frontmatter |
| CSV | .csv | tables, row/column metadata |
| JSON | .json | pretty-print, table from list-of-dicts |
| Image | .png, .jpg, .jpeg, .tiff, .bmp, .webp | metadata, OCR text |
| EPUB | .epub | text, tables, images, metadata |
| YouTube | — | transcript, timestamps, chapters |
| Notion | — | pages, databases, 14 block types |
| Audio | .mp3, .wav, .flac, .m4a, .ogg, .wma, .aac, .opus | transcribed text, segments |
| Video | .mp4, .avi, .mov, .mkv, .webm, .flv, .wmv | key-frame images, transcript |

## Features

### Chunking (6 strategies)
fixed_size, by_page, by_heading, semantic, by_token, sentence_window

### AI Providers (10)
OpenAI, Anthropic, Gemini, Ollama, Azure, Bedrock, Groq, Together, DeepSeek, Mistral

### RAG
- Hybrid search (dense + BM25)
- Metadata filtering
- ChromaDB / FAISS vector stores
- Contextual compression
- Extract-and-index pipeline
- Auto-RAG (zero-config pipeline)
- Hierarchical / RAPTOR chunking
- Multi-modal RAG (text + tables + images)
- Citation engine (auto-cite with `[N]` markers)

### Structured Extraction
```python
from pydantic import BaseModel
from runeextract import extract_structured

class Invoice(BaseModel):
    invoice_number: str
    total: float

result = extract_structured("invoice.pdf", Invoice)
```

### Agent Tools
- MCP Server tools (extract, search, crawl)
- LangChain `RuneExtractLoader`
- LlamaIndex `RuneExtractReader`
- CrewAI `RuneExtractTool`
- AutoGen `autogen_extract_tool`
- Function calling / structured output
- Streaming responses
- Rate limiter
- Cost tracking
- Query expansion (HyDE + multi-query)
- Batch processing

### Document Processing Pipeline
```python
from runeextract import Pipeline

Pipeline([
    Pipeline.extract(),
    Pipeline.chunk(strategy="semantic", size=500),
    Pipeline.embed(provider="openai"),
    Pipeline.store(collection="docs"),
]).run(["file1.pdf", "file2.pdf"])
```

### Media
- Audio transcription (Whisper / transformers)
- Video frame extraction + transcription

### File Sync & Watching
```python
from runeextract import DirectoryWatcher, FileSync

watcher = DirectoryWatcher("~/docs", patterns=["*.pdf", "*.docx"])
events = watcher.poll()  # created, modified, deleted

FileSync("~/source", "~/backup").sync(patterns=["*.md"])
```

### Web Crawling
```python
from runeextract import smart_crawl, parse_sitemap, parse_feed

results = smart_crawl("https://example.com", max_pages=20)
urls = parse_sitemap("https://example.com/sitemap.xml")
entries = parse_feed("https://example.com/feed.xml")
```

### Document Diff
```python
from runeextract import diff_documents, compare_files

result = diff_documents("old version", "new version")
print(result.summary())  # "Changes: 3 total (1 added, 1 removed, 1 modified)"
```

### Layout-Aware Parsing
```python
from runeextract import parse_layout, get_reading_order

elements = parse_layout("## Heading\n\nParagraph text", source_type="text")
ordered = get_reading_order(elements)
```

### ONNX On-Device Embeddings
```python
from runeextract import get_onnx_embedding

model = get_onnx_embedding()
vectors = model.embed(["text to embed"])
```

### Cloud Storage
```python
from runeextract import get_storage_connector

s3 = get_storage_connector("s3", bucket="my-bucket")
data = s3.read("documents/report.pdf")
```

### Quality
- Document quality scoring (0-100)
- PII redaction
- Differential privacy engine
- Secret scanning (30+ patterns)
- Magic-byte format detection
- Per-file cache with TTL
- Memory profiling

## Benchmark Suite
```bash
python -c "from runeextract.benchmarks import run_all_benchmarks; runner = run_all_benchmarks('test.pdf'); print(runner.summary())"
```

## More Examples

See the [docs/](docs/) directory for detailed guides:

| Document | Description |
|----------|-------------|
| [features.md](docs/features.md) | Complete feature reference |
| [usage.md](docs/usage.md) | Installation, CLI, config, plugins |
| [examples.md](docs/examples.md) | Rich code examples for every feature |
| [ai.md](docs/ai.md) | AI provider details, cost tracking, rate limiting |
| [rag.md](docs/rag.md) | Chunking, search, vector stores, question answering |

### CLI

```bash
runeextract file.pdf --ocr --chunking semantic --tree
runeextract file.pdf --summary --keywords
runeextract https://youtube.com/watch?v=... --youtube-format transcript
runeextract ./docs --watch
runeextract file.pdf --ask "What is this about?"
runeextract file.pdf --json --output-dir ./output
```

## Test Suite

```bash
pip install -e ".[dev]"
pytest runeextract/tests/   # 569 tests
```

## Architecture

```
File/URL → Router (magic-byte detection) → Extractor → Document (unified schema)
                                                          ↓
                                                    Chunking → Search → RAG
                                                          ↓
                                                    AI → Summary, Entities, Q&A
                                                          ↓
                                                    Export → JSON, Markdown, Vector DB

NEW MODULES:
  Structured   → Pydantic schema extraction via LLM
  Citation     → Auto-cite claims with [N] markers
  Web          → Smart crawler, sitemap, RSS/Atom
  Transform    → DAG pipeline (9 step types)
  Sync         → File watching, sync, batch extraction
  Agent        → MCP, LangChain, LlamaIndex, CrewAI, AutoGen
  Layout       → Bounding boxes, columns, reading order
  Diff         → Version comparison, change tracking
  Embeddings   → ONNX on-device embedding models
  Storage      → S3, GCS, Azure Blob connectors
  Benchmarks   → Performance comparison vs competitors
```

## License

MIT — see [LICENSE](LICENSE).

## Why RuneExtract?

Instead of learning PyMuPDF, python-docx, BeautifulSoup, EasyOCR, Whisper, OpenCV, etc.:

```python
extract(anything)
```

That simplicity is the entire product.
