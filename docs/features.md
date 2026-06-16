# Features

## Extraction (14+ Formats)

| Format | Extensions | Extracted Content |
|--------|-----------|-------------------|
| PDF | .pdf | text, tables, images, metadata, scanned-page OCR |
| DOCX | .docx, .doc | paragraphs, tables, images, metadata, image OCR |
| PPTX | .pptx, .ppt | slides, tables, images, speaker notes, metadata |
| XLSX | .xlsx, .xls | worksheets, tables, multiple sheets, metadata |
| HTML | .html, .htm | headings, paragraphs, tables, links, meta tags, images |
| Markdown | .md, .markdown | headings, lists, code blocks, tables, frontmatter |
| CSV | .csv | tables, text, row/column metadata |
| JSON | .json | pretty-print, table from list-of-dicts |
| Image | .png, .jpg, .jpeg, .tiff, .tif, .bmp, .webp | metadata, OCR text (optional) |
| EPUB | .epub | text, tables, images, metadata (title, author, etc.) |
| YouTube | — | transcript, timestamps, chapters, metadata |
| Notion | — | pages, databases, 14 block types, async |
| Audio | .mp3, .wav, .flac, .m4a, .ogg, .wma, .aac, .opus | transcribed text, segments, duration |
| Video | .mp4, .avi, .mov, .mkv, .webm, .flv, .wmv | key-frame images, transcribed audio text |

## Chunking Strategies

| Strategy | Description |
|----------|-------------|
| `fixed_size` | Split by character count with configurable overlap |
| `by_page` | Use metadata page breaks as chunk boundaries |
| `by_heading` | Split on markdown headings or underline headings |
| `semantic` | Split at paragraph boundaries (\n\n) |
| `by_token` | Split by token count using tiktoken |
| `sentence_window` | Split on sentence boundaries, group into windows with overlap |

## AI Providers (10)

| Provider | Env Variable | Features |
|----------|-------------|----------|
| OpenAI | `OPENAI_API_KEY` | Chat, embeddings, function calling, streaming |
| Anthropic | `ANTHROPIC_API_KEY` | Chat, streaming |
| Google Gemini | `GEMINI_API_KEY` | Chat |
| Ollama (local) | — | Chat, embeddings (local) |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` | Chat, function calling, streaming |
| AWS Bedrock | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` | Chat (Anthropic models) |
| Groq | `GROQ_API_KEY` | Chat, function calling, streaming |
| Together AI | `TOGETHER_API_KEY` | Chat, function calling, streaming |
| DeepSeek | `DEEPSEEK_API_KEY` | Chat, function calling, streaming |
| Mistral AI | `MISTRAL_API_KEY` | Chat, function calling, streaming |

## RAG Features

- **Hybrid search**: Dense (cosine similarity) + BM25 sparse fusion
- **Metadata filtering**: Pre-filter chunks by exact metadata match
- **Vector stores**: ChromaDB, FAISS integration
- **Contextual compression**: Retrieve → rerank → LLM-compress pipeline
- **Extract-and-index**: Single-call extraction + chunking + vector indexing
- **Multi-modal messages**: Base64 images in OpenAI vision format

## Agent & Developer Tools

- **Function calling**: `call_with_tools()` for structured output
- **Streaming**: `_call_stream()` / `call_stream_async()` token-by-token
- **Rate limiter**: Token-bucket for API call throttling
- **Cost tracking**: Per-call token usage and USD cost estimates
- **Query expansion**: HyDE + multi-query generation
- **Batch processing**: Concurrent LLM calls via ThreadPoolExecutor

## Quality & Validation

- **Document quality scoring**: `score_quality()` — text density, readability, structure, completeness, OCR confidence (0-100)
- **PII redaction**: `redact_pii()` — AI-powered PII removal
- **Magic-byte detection**: Content-based format detection without extension
- **Per-file cache invalidation**: TTL-based caching with per-file granularity

## Web Crawling

- **`extract_crawl()`**: Breadth-first web crawler
- Same-domain filtering
- robots.txt respect
- Configurable delay and max pages
