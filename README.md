# RuneExtract

**One extraction API for every document type.** v0.7.0

RuneExtract is a universal document extraction library that provides a single, consistent API for extracting content from any file type — PDFs, DOCX, HTML, images, audio, video, YouTube, Notion, and more.

## 🚀 5 Game-Changing Features for RAG Developers

**Zero-config RAG that developers love.** RuneExtract eliminates weeks of boilerplate with production-ready features:

```python
from runeextract import auto_rag

# ONE LINE - Production RAG with adaptive intelligence, live sync, citations, 
# multi-modal understanding, and built-in safeguards
rag = auto_rag(
    "./documents/",
    intelligence="adaptive",    # 🎯 Self-tuning retrieval
    watch=True,                 # ⚡ Auto-sync on file changes
    multimodal=True,            # 👁️  Understand charts & images
    safe_mode=True,             # 🛡️  Cost limits, secret scanning
    cost_limit=10.00            # 💰 Hard budget cap
)

# Query with automatic citations and provenance
result = rag.query("What are the key findings?", cite=True)
print(result.answer)
# "The study found a 23% improvement [1] using the new methodology [2]."

print(f"Cost: ${result.cost:.4f} | Confidence: {result.confidence:.2%}")
# Cost: $0.0234 | Confidence: 87%
```

### Why Developers Choose RuneExtract

| Feature | Pain Solved | Time Saved |
|---------|-------------|-----------|
| **🎯 Adaptive Intelligence** | Hours of parameter tuning → Zero config | 4-8 hours |
| **⚡ Live Document Sync** | Manual re-indexing → Auto-sync background | 30 min/update |
| **📚 Citation Engine** | Custom citation code → One parameter | 2-3 days |
| **👁️  Multi-Modal RAG** | Manual image extraction → Automatic | 1-2 weeks |
| **🛡️  Production Safeguards** | Security hardening → Built-in | 2-3 weeks |

**Total: 4-6 weeks saved per RAG project.** [Read the full guide →](docs/GAME_CHANGING_FEATURES.md)

---

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

# Streaming AI
for chunk in doc.ask_stream("Summarize this document."):
    print(chunk, end="", flush=True)

# Multi-turn conversation
chat = doc.chat()
answer1 = chat.ask("What is section 3 about?")
answer2 = chat.ask("Can you elaborate?")  # remembers context

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

### Chunking (7 strategies)
fixed_size, by_page, by_heading, semantic, by_token, sentence_window, hierarchical (RAPTOR)

### AI Providers (12)
OpenAI, OpenRouter, Anthropic, Gemini, Ollama, Azure, Bedrock, Groq, Together, DeepSeek, Mistral, Local (transformers)

### Streaming & Multi-turn Chat
- **Streaming AI**: `doc.ask_stream()` yields tokens as they arrive from the LLM
- **Chat Sessions**: `doc.chat()` creates a `ChatSession` with conversation memory
- Custom system prompts, manual message injection, RAG context integration

### RAG
- **Hybrid search OOTB** (dense + BM25 with RRF fusion, enabled by default)
- **Source grounding** (char offsets through Chunk → ChunkWithScore → Citation)
- **Auto query rewriter** (detects question type, auto-enables HyDE/MultiQuery)
- **Domain templates** (financial/legal/medical/academic presets in `auto_rag(..., domain=)`)
- **Multi-level caching** (LRU+TTL for embeddings, search, answers — wired into AutoRAG)
- **Query router** (intent classification, filter extraction, query decomposition in `auto_rag(..., query_router=True)`)
- **Context packer** (token-budget-aware chunk packing via `rag.query(..., max_tokens=2000)`)
- **Adaptive hybrid search** (query-aware dense/sparse weight tuning)
- Metadata filtering
- ChromaDB / FAISS vector stores
- Contextual compression
- Extract-and-index pipeline
- Auto-RAG (zero-config pipeline)
- Hierarchical / RAPTOR chunking (with parent-child links)
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
- **MCP Server CLI** (`runeextract-mcp`) with `extract_url`, `ask_documents`, `chunk_document` tools
- **LangGraph** `RuneExtractGraphTool`, `RuneExtractSearchTool`, `RuneExtractAskTool`
- **OpenAI Agents SDK** `rune_extract_function_tool()`, `rune_extract_search_tool()`
- **PydanticAI** `RuneExtractAITool`, `RuneExtractSearchAITool`
- LangChain `RuneExtractLoader`
- LlamaIndex `RuneExtractReader`
- CrewAI `RuneExtractTool`
- AutoGen `autogen_extract_tool`
- Function calling / structured output
- Streaming responses (`_call_stream`, `ask_stream`)
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

### Architecture
The provider system has been modularized into a plugin-style registry:
- `processors/providers/` — 6 handler modules: `openai_compat` (OpenAI, OpenRouter, Azure, Ollama, Groq, Together, DeepSeek, Mistral), `anthropic`, `gemini`, `bedrock`, `local`
- The `Document` class has been refactored into focused modules: `models/types.py`, `models/chunking.py`, `models/chat_session.py`, `models/document.py`
- `AIProcessor` reduced from 1135 to 489 lines — all provider-specific calls delegated to the registry

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
| [troubleshooting.md](docs/TROUBLESHOOTING.md) | Common issues and solutions |

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
pytest runeextract/tests/   # 868 tests
```

## Architecture

```
File/URL → Router (magic-byte detection) → Extractor → Document (unified schema)
                                                          ↓
                                              ┌───── Chunking ─────────┐
                                              │ fixed_size, by_page,   │
                                              │ by_heading, semantic,  │
                                              │ by_token, sent_window  │
                                              │ hierarchical (RAPTOR)  │
                                              └─────────┬──────────────┘
                                                        ↓
                                              ┌──── Search & RAG ──────┐
                                              │ dense / sparse / hybrid│
                                              │ ChromaDB / FAISS       │
                                              │ Contextual compression │
                                              │ Auto-RAG, HyDE,        │
                                              │ Multi-query expansion   │
                                              └─────────┬──────────────┘
                                                        ↓
                                              ┌──── AI & Chat ─────────┐
                                              │ Summary, Entities, QA  │
                                              │ Streaming (ask_stream) │
                                              │ Multi-turn (ChatSession)│
                                              │ Vision (describe_image)│
                                              │ Structured extraction  │
                                              └─────────┬──────────────┘
                                                        ↓
                                              ┌──── Export ────────────┐
                                              │ JSON, Markdown, Dict   │
                                              │ Vector DB, LangChain,  │
                                              │ LlamaIndex, CrewAI,    │
                                              │ MCP Server, AutoGen    │
                                              └────────────────────────┘
```

### Module Map

| Directory | Purpose |
|-----------|---------|
| `core/` | Router, extractor base, cache, registry, streaming, async |
| `models/` | Document, Chunk, Table, Image, chunking, ChatSession |
| `processors/` | AIProcessor (reduced), OCR, provider registry |
| `providers/` | 6 handler modules for 12 providers |
| `extractors/` | 14 format extractors |
| `rag/` | Chunking, search, vector stores, auto-RAG, evaluation |
| `transform/` | DAG pipeline with 9 step types |
| `vision/` | Image/chart/figure analysis |
| `web/` | Crawler, sitemap, RSS/Atom feed |
| `sync/` | File watching, directory sync |
| `agent/` | MCP server, LangChain, LlamaIndex, CrewAI, AutoGen |
| `layout/` | Bounding boxes, reading order |
| `diff/` | Document comparison, change tracking |
| `embeddings/` | ONNX on-device embedding models |
| `storage/` | S3, GCS, Azure Blob connectors |
| `benchmarks/` | Performance benchmarks vs competitors |
| `dedup/` | MinHash, LSH, embedding deduplication |
| `server/` | WebSocket extraction server |

## License

MIT — see [LICENSE](LICENSE).

## Why RuneExtract?

Instead of learning PyMuPDF, python-docx, BeautifulSoup, EasyOCR, Whisper, OpenCV, etc.:

```python
extract(anything)
```

That simplicity is the entire product.
