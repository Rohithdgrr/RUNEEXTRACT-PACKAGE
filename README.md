# RuneExtract

**One extraction API for every document type.** v0.4.0

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
| **Audio** | .mp3, .wav, .flac, .m4a, .ogg, .wma, .aac, .opus | transcribed text, segments |
| **Video** | .mp4, .avi, .mov, .mkv, .webm, .flv, .wmv | key-frame images, transcript |

## Features

### Chunking (6 strategies)
fixed_size, by_page, by_heading, semantic, by_token, **sentence_window**

### AI Providers (10)
OpenAI, Anthropic, Gemini, Ollama, **Azure**, **Bedrock**, **Groq**, **Together**, **DeepSeek**, **Mistral**

### RAG
- Hybrid search (dense + BM25)
- Metadata filtering
- ChromaDB / FAISS vector stores
- Contextual compression
- Extract-and-index pipeline

### Agent Tools
- Function calling / structured output
- Streaming responses
- Rate limiter
- Cost tracking
- Query expansion (HyDE + multi-query)
- Batch processing

### Media
- Audio transcription (Whisper / transformers)
- Video frame extraction + transcription

### Quality
- Document quality scoring (0-100)
- PII redaction
- Magic-byte format detection
- Per-file cache with TTL

### Web Crawling
```python
from runeextract import extract_crawl
docs = extract_crawl("https://example.com", max_pages=20)
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
pytest runeextract/tests/   # 184 tests
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
```

## License

MIT — see [LICENSE](LICENSE).

## Why RuneExtract?

Instead of learning PyMuPDF, python-docx, BeautifulSoup, EasyOCR, Whisper, OpenCV, etc.:

```python
extract(anything)
```

That simplicity is the entire product.
