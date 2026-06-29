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

## Streaming AI & Multi-Turn Chat

- **Streaming**: `doc.ask_stream()` yields tokens as they arrive, `AIProcessor._call_stream()` for direct use
- **Chat Sessions**: `doc.chat()` creates `ChatSession` with conversation memory for multi-turn dialogue
- **Custom system prompts**: Per-chat-system-prompt support
- **Manual message injection**: `add_user_message()`, `add_assistant_message()` for pre-seeding conversations

## AI Providers (12)

| Provider | Env Variable | Features |
|----------|-------------|----------|
| OpenAI | `OPENAI_API_KEY` | Chat, embeddings, function calling, streaming |
| OpenRouter | `OPENAI_API_KEY` + `OPENAI_BASE_URL` | Chat, function calling, streaming (200+ models) |
| Anthropic | `ANTHROPIC_API_KEY` | Chat |
| Google Gemini | `GEMINI_API_KEY` | Chat |
| Ollama (local) | — | Chat, embeddings (local) |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` | Chat, function calling, streaming |
| AWS Bedrock | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` | Chat (Anthropic models) |
| Groq | `GROQ_API_KEY` | Chat, function calling, streaming |
| Together AI | `TOGETHER_API_KEY` | Chat, function calling, streaming |
| DeepSeek | `DEEPSEEK_API_KEY` | Chat, function calling, streaming |
| Mistral AI | `MISTRAL_API_KEY` | Chat, function calling, streaming |
| Local (transformers) | — | Chat, embeddings (on-device) |

## RAG Features

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

## RAG Features (Tier 2 — 9 Game-Changing Features)

| # | Feature | Description | Wiring |
|---|---------|-------------|--------|
| 6 | **Smart Query Routing** | Intent-based routing to specialized RAG pipelines | `auto_rag(..., routing_rags={...})` |
| 7 | **Semantic Caching** | Embedding-based cache saves 60%+ on repeat queries | `auto_rag(..., semantic_cache=True)` |
| 8 | **Analytics Dashboard** | Query metrics, time series, latency tracking, CSV export | `auto_rag(..., analytics=True)` |
| 9 | **A/B Experiments** | Multi-variant testing with user bucketing, statistical significance | `auto_rag(..., experiment_config={...})` |
| 10 | **Multi-Language** | Auto-detect language, translate queries, cross-lingual search | `auto_rag(..., multi_language=True)` |
| 11 | **RBAC** | Role-based access control, field-level redaction, audit logging | `auto_rag(..., rbac=True)` |
| 12 | **Streaming RAG** | Progressive refinement with 5-stage streaming pipeline | `auto_rag(..., streaming=True)` |
| 13 | **RAG-as-a-Service API** | FastAPI REST API with auth, rate limiting, streaming, metrics | `rag.create_api_server()` / `rag.serve()` |
| 14 | **Chain-of-Thought** | Multi-step reasoning, decomposition, self-correction | `auto_rag(..., reasoning=True)` |

## v0.8.0 New Features

### Phase 1.5: Ecosystem

| Feature | Description | Usage |
|---------|-------------|-------|
| **MCP Server CLI** | Model Context Protocol server for Claude Desktop + agent frameworks | `runeextract-mcp` or `extract(...mcp_server=True)` |
| **LangGraph Integration** | `RuneExtractGraphTool`, `RuneExtractSearchTool`, `RuneExtractAskTool` | `from runeextract.agent import RuneExtractGraphTool` |
| **OpenAI Agents SDK** | Function tool for the OpenAI Agents SDK | `from runeextract.agent import rune_extract_function_tool` |
| **PydanticAI Integration** | `RuneExtractAITool`, `RuneExtractSearchAITool` | `from runeextract.agent import RuneExtractAITool` |
| **Parent-Child Chunking** | RAPTOR-style hierarchical chunks with parent links | `doc.hierarchical_chunks(child_size=300, parent_size=1500)` |

### Phase 0: Foundation

| Feature | Description | Default |
|---------|-------------|---------|
| **Source Grounding** | Char offsets propagated through Chunk → ChunkWithScore → Citation | Always on |
| **Hybrid Search OOTB** | Dense + BM25 with RRF fusion enabled by default | `hybrid_search=True` |
| **Auto Query Rewriter** | `QueryAnalyzer` auto-enables HyDE/MultiQuery based on question type | `auto_query=True` |

### Phase 1: Quality & Trust

| Feature | Description | Integration |
|---------|-------------|-------------|
| **Domain Templates** | Pre-configured presets (financial, legal, medical, academic) | `auto_rag(..., domain="financial")` |
| **Embedding Auto-Selection** | Resolve "fast" / "balanced" / "accurate" to concrete models | `resolve_embedding("fast")` |
| **Multi-Level Caching** | Three-level LRU+TTL cache (embeddings, search, answers) | Wired into `AutoRAG._retrieve()` |

### Phase 2: Growth

| Feature | Description | Integration |
|---------|-------------|-------------|
| **Query Router** | Intent classification, filter extraction, query decomposition | `auto_rag(..., query_router=True)` |
| **Adaptive Hybrid Search** | Query-aware dense/sparse weight tuning | `HybridSearch.compute_weights()` in `_retrieve()` |
| **Context Packer** | Token-budget-aware chunk packing | `rag.query(..., max_tokens=2000)` |

## Web Crawling

- **`extract_crawl()`**: Breadth-first web crawler
- Same-domain filtering
- robots.txt respect
- Configurable delay and max pages
