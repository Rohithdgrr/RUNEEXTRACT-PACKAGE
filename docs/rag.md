# RAG (Retrieval-Augmented Generation)

## Chunking

Documents must be chunked before RAG operations:

```python
from runeextract import extract
from runeextract.models.document import ChunkingStrategy

doc = extract("report.pdf")

# Available strategies
doc.chunks(strategy=ChunkingStrategy.SENTENCE_WINDOW, size=5, overlap=1)
doc.chunks(strategy=ChunkingStrategy.FIXED_SIZE, size=1000, overlap=100)
doc.chunks(strategy=ChunkingStrategy.SEMANTIC, size=1000)
doc.chunks(strategy=ChunkingStrategy.BY_PAGE)
doc.chunks(strategy=ChunkingStrategy.BY_HEADING)
doc.chunks(strategy=ChunkingStrategy.BY_TOKEN, size=500, overlap=50)
```

Chunks are cached — calling `chunks()` again with the same params returns the cached result.

## Search

### Dense (default)

```python
results = doc.search("machine learning", top_k=5)
for chunk, score in results:
    print(f"[{score:.4f}] {chunk.text[:100]}...")
```

### Hybrid (dense + BM25)

Requires `rank_bm25` package (`pip install rank_bm25`):

```python
results = doc.search("deep learning", mode="hybrid", top_k=5)
```

### Sparse (BM25 only)

```python
results = doc.search("neural networks", mode="sparse")
```

### Metadata filtering

```python
# Only search chunks from page 1
results = doc.search("introduction", metadata_filter={"page_number": 1})

# Multiple metadata filters
results = doc.search("results", metadata_filter={
    "page_number": 5,
    "section": "methodology"
})
```

### Retrieve (wrapper)

```python
chunks = doc.retrieve("What is the conclusion?", top_k=3)
```
Returns just the chunk texts (no scores).

## Vector Stores

### ChromaDB

```python
doc.to_chromadb(
    collection_name="my_docs",
    persist_directory="./chroma_db",
)

# Query directly
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("my_docs")
results = collection.query(query_texts=["machine learning"], n_results=5)
```

### FAISS

```python
doc.to_faiss(index_path="./faiss_index")

# Query directly
import faiss
import numpy as np
index = faiss.read_index("./faiss_index")
# ... query with embeddings
```

### Extract and index (one-call)

```python
from runeextract import extract_and_index

doc = extract_and_index(
    "research_paper.pdf",
    store="chromadb",
    collection_name="papers",
    chunking_strategy="semantic",
    chunk_size=1000,
)
```

## Question Answering

```python
doc = extract("company_policy.pdf", chunking_strategy="semantic")
answer = doc.ask("What is the remote work policy?")
print(answer)
```

This performs: chunk → search → retrieve → LLM answer generation.

### Streaming Question Answering

```python
for chunk in doc.ask_stream("What is the remote work policy?"):
    print(chunk, end="", flush=True)
```

### Multi-Turn Chat with RAG

```python
# Create a chat session with document context
chat = doc.chat(system_prompt="You are a policy expert.")

# Conversation remembers both document context and chat history
answer1 = chat.ask("What is the vacation policy?")
answer2 = chat.ask("How does that compare to sick leave?")  # remembers context
answer3 = chat.ask("Can you summarize both?")  # builds on previous answers
```

## Contextual Compression

```python
doc = extract("long_article.pdf", chunking_strategy="sentence_window", size=5)
compressed = doc.compress("key findings", top_k=10)
print(compressed)
```

Pipeline: retrieve chunks → rerank → extract query-relevant sentences → stay within token budget.

## Multi-modal Messages

```python
doc = extract("presentation.pptx", images=True)
messages = doc.to_openai_messages(
    system_message="Describe the content of this document.",
    include_images=True,  # embeds images as base64
)
```

The output format is compatible with OpenAI vision API:

```json
[
    {"role": "system", "content": "Describe..."},
    {"role": "user", "content": [
        {"type": "text", "text": "slide content..."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]}
]
```

## Query Expansion

Use AI to generate better search queries:

```python
from runeextract.processors.ai import AIProcessor

ai = AIProcessor()

# Generate multiple query variations
queries = ai.expand_query("How do transformers work?", n_queries=3)
for q in queries:
    results = doc.search(q)

# Generate hypothetical document for embedding
hypothetical = ai.hyde("Explain the attention mechanism")
hyde_embedding = ai.embed([hypothetical])
```

## Document Quality Scoring

```python
doc = extract("invoice.pdf")
scores = doc.score_quality()
print(f"Overall quality: {scores['overall']}/100")

# Per-dimension scores
print(f"Text density: {scores['text_density']}")
print(f"Readability: {scores['readability']}")
print(f"Structure: {scores['structure']}")
print(f"Completeness: {scores['completeness']}")
print(f"OCR confidence: {scores['ocr_confidence']}")
```

## Export Formats

```python
doc = extract("data.docx")

# To LlamaIndex
llama_docs = doc.to_llamaindex_documents()

# To pandas DataFrame
df = doc.tables[0].to_dataframe()

# To OpenAI messages (for chat completion)
messages = doc.to_openai_messages(system_message="Analyze this.")

# To Markdown
text = doc.to_markdown()

# To JSON
data = doc.to_json()
```

## Auto-RAG Pipeline

Zero-configuration end-to-end RAG pipeline that ingests, chunks, indexes, retrieves, and answers.

### Quick-start

```python
from runeextract import auto_rag

rag = auto_rag("report.pdf")
result = rag.query("What are the key findings?")
print(result.answer)
# Access citations, confidence, latency
for c in result.citations:
    print(f"  [{c.source}] {c.text[:100]}")
```

### One-liner with `instant_rag`

Smarter defaults (FAISS, auto-embedding, caching):

```python
from runeextract import instant_rag

rag = instant_rag("report.pdf", domain="financial")
result = rag.query("Net income?")
```

### Options

```python
rag = auto_rag(
    "docs/",
    embedding="openai:text-embedding-3-large",
    vector_store="faiss",          # "chromadb" or "faiss"
    chunking="auto",               # auto-detected based on content
    chunk_size=1000,
    chunk_overlap=100,
    reranker="cross-encoder/ms-marco-MiniLM-L-6-v2",
    llm="openai:gpt-4o-mini",
)
```

### Query modes

```python
# Multi-query expansion — generates 3 query variants and fuses results
result = rag.query("What is the revenue?", multi_query=True)

# HyDE — generates a hypothetical document for embedding
result = rag.query("Explain the policy.", hyde=True)

# Answer length control
result = rag.query("Summarize.", answer_length="short")  # "short" | "medium" | "long"

# Token budget for context window
result = rag.query("Key points?", max_tokens=2000)
```

### Domain Templates (Phase 1, wired into AutoRAG v0.8.0+)

Pre-configured settings for common document types — **now wired directly into `auto_rag()`**:

```python
# Auto-applies domain-optimized chunking, embedding, reranker
rag = auto_rag("report.pdf", domain="financial")

# Or via AutoRAG directly
rag = AutoRAG(domain="medical")
rag.ingest("clinical_trial.pdf")
```

Manual access still available:

```python
from runeextract.rag import DomainTemplates, DomainConfig

# Four built-in presets
financial = DomainTemplates.get("financial")   # by_heading, 800chunks, reranker
legal = DomainTemplates.get("legal")           # by_heading, 1200chunks
medical = DomainTemplates.get("medical")       # sentence_window, 600chunks
academic = DomainTemplates.get("academic")     # hierarchical, 1000chunks

# Register custom presets
DomainTemplates.register("custom", DomainConfig(
    chunking="fixed_size", chunk_size=500,
    reranker="cross-encoder/ms-marco-MiniLM-L-6-v2",
))
```

Because `resolve_embedding()` is called in `AutoRAG.__init__()`, passing `embedding="fast"` or `embedding="balanced"` is automatically resolved to the concrete model string.

### Multi-Level Caching (Phase 1, wired into AutoRAG v0.8.0+)

Three LRU+TTL cache levels for embeddings, search results, and answers — **automatically used inside AutoRAG**:

```python
# AutoRAG uses RAGCache internally for:
#   - Embedding caching (avoids re-embedding repeated queries)
#   - Search result caching (avoids re-querying vector store)
rag = AutoRAG(cache_maxsize=500)  # default

# Check cache stats
stats = rag.cache_stats()
print(stats["rag_cache"]["embedding_cache_size"])
```

Standalone usage:

```python
from runeextract.rag import RAGCache

cache = RAGCache(maxsize=1000)

# Embedding cache (TTL: 1 hour)
cache.put_embedding("some text", [0.1, 0.2, ...])
emb = cache.get_embedding("some text")

# Search cache (TTL: 5 minutes)
cache.put_search("query?", top_k=5, chunks)
result = cache.get_search("query?", top_k=5)

# Answer cache (TTL: 10 minutes)
cache.put_answer("question?", "answer")
ans = cache.get_answer("question?")

# Clear all
cache.invalidate()
```

### Query Router (Phase 2, wired into AutoRAG v0.8.0+)

Classifies intent, extracts metadata filters, and decomposes complex queries — **now wired into AutoRAG**:

```python
# Enable in AutoRAG — auto-classifies queries, extracts metadata filters
rag = AutoRAG(query_router=True)
result = rag.query("revenue in 2024")  # auto-extracts {"year": "2024"}

# Comparative/analytical queries are auto-decomposed into sub-queries
rag = AutoRAG(query_router=True)
result = rag.query("Compare Q1 and Q2 results")
# → runs sub-queries ["What is Q1 revenue?", "What is Q2 revenue?"]
# → fuses retrieval results
```

Standalone usage:

```python
from runeextract.rag import QueryRouter, QueryIntent

router = QueryRouter()

# Intent classification
intent = router.classify("Compare Q1 and Q2 results")  # QueryIntent.COMPARATIVE
intent = router.classify("Why did revenue decline?")    # QueryIntent.ANALYTICAL

# Metadata filter extraction
filters = router.extract_filters("revenue in 2024")    # {"year": "2024"}
filters = router.extract_filters('results by "John Smith"')  # {"author": "John Smith"}

# Full decomposition
dq = router.decompose("Compare Q1 and Q2 revenue")
print(dq.intent, dq.sub_queries, dq.metadata_filter)
```

### Source Grounding (v0.8.0+)

Every chunk and citation carries character offsets:

```python
result = rag.query("What is the conclusion?")
for c in result.citations:
    print(f"  chars {c.char_start}-{c.char_end}: {c.text[:50]}...")
```

The chain: **Chunk.start_index → ChromaDB metadata → ChunkWithScore.char_start → Citation.char_start**.

### Hybrid Search (OOTB, v0.8.0+)

Hybrid search (Dense + BM25 with Reciprocal Rank Fusion) is **enabled by default**:

```python
rag = AutoRAG()  # hybrid_search=True by default
result = rag.query("machine learning")
```

BM25 is computed against all indexed chunks and fused with dense vector scores via RRF. Disable with `AutoRAG(hybrid_search=False)`.

### Auto Query Rewriter (v0.8.0+)

The `QueryAnalyzer` automatically detects question type and enables HyDE + MultiQuery:

```python
rag = AutoRAG()  # auto_query=True by default
# Analytical questions auto-enable HyDE + MultiQuery
result = rag.query("Why did the experiment fail?")

# Simple factual queries skip expansion
result = rag.query("What is the capital of France?")
```

You can still override manually:

```python
result = rag.query("question", hyde=True, multi_query=False)
```

### Hybrid Search (Phase 2, wired into AutoRAG v0.8.0+)

Adaptive dense + sparse retrieval with zero-dependency BM25 — **weights are now automatically tuned based on query analysis** via `HybridSearch.compute_weights()`:

```python
# In AutoRAG, hybrid search uses adaptive weights internally
rag = AutoRAG(hybrid_search=True)  # default
result = rag.query("What is Q1 revenue by region?")
# → adaptive dense/sparse weighting based on query lexical density
```

Standalone usage:

```python
from runeextract.rag import HybridSearch, ChunkWithScore

hs = HybridSearch(dense_fn=lambda q: [0.1, 0.2] * 128, chunks=chunks)

# Adaptive weights based on query analysis
result = hs.search("What is Q1 revenue by region?", top_k=5)
print(result.dense_weight, result.sparse_weight)  # e.g. 0.4, 0.6
print(result.query_analysis)  # {"lexical_density": ..., "term_count": ...}
```

### Context Packer (Phase 2, wired into AutoRAG v0.8.0+)

Intelligently fits chunks into an LLM's token budget — **pass `max_tokens` directly to `rag.query()`**:

```python
rag = AutoRAG()
result = rag.query("Summarize the report", max_tokens=2000)
# → chunks are packed into ~2000 token budget before LLM call
```

Standalone usage:

```python
from runeextract.rag import ContextPacker

packer = ContextPacker(max_tokens=2000)

# Strategies: "sorted" | "compressed" | "structured"
packed = packer.pack(chunks, query="What is revenue?", strategy="sorted")
print(packed.text, packed.chunks_used, packed.total_tokens)

packed = packer.pack(chunks, query, strategy="compressed")   # summarise low-score
packed = packer.pack(chunks, query, strategy="structured")   # group by source
```

### RobustRAG with Fallbacks (Phase 4)

Graceful degradation when primary retrieval or LLM fails:

```python
from runeextract.rag import RobustRAG, FallbackStrategy
from runeextract import auto_rag

base = auto_rag("docs.pdf")
rag = RobustRAG(base)

# Automatic fallback: primary → keyword search → LLM-only
result = rag.query("What is X?")
print(rag.fallback_used)  # "primary_retriever" | "keyword_fallback" | "llm_only"

# Custom strategies with retries
rag = RobustRAG(base, strategies=[
    FallbackStrategy("primary_retriever", max_retries=2),
    FallbackStrategy("keyword_fallback", max_retries=1),
    FallbackStrategy("llm_only"),
])
```

### Confidence Scoring (Phase 4)

Multi-factor answer confidence:

```python
from runeextract.rag import ConfidenceScorer

scorer = ConfidenceScorer()
factors = scorer.score(chunks, "Revenue grew 20%.", "What happened?")

print(factors.retrieval_score)    # distribution quality
print(factors.source_diversity)   # unique sources / total chunks
print(factors.chunk_relevance)    # word overlap with question
print(factors.faithfulness)       # lexical or LLM-judged support
print(factors.overall)            # weighted combination

# With LLM judge for faithfulness
scorer = ConfidenceScorer(llm_judge=lambda a, c: 0.9)
```

### RAG Debugger (Phase 4)

Step-by-step execution trace of every pipeline stage:

```python
from runeextract.rag import RAGDebugger
from runeextract import auto_rag

rag = auto_rag("report.pdf")
debugger = RAGDebugger(rag)

trace = debugger.trace("What is the revenue?", multi_query=True)

# Print formatted trace
debugger.print_trace(trace)

# Export to dict for analysis
d = debugger.trace_to_dict(trace)

# Access individual stages
print(trace.stages)            # {"query_expansion": 12.3, "retrieval": 45.6, ...}
print(trace.query_variants)    # expanded query list
print(trace.retrieved_chunks)  # before reranking
print(trace.reranked_chunks)   # after reranking
print(trace.errors)            # any failures per stage
```

### RAG Evaluation (Phase 4)

```python
from runeextract.rag import RAGEvaluator
from runeextract import auto_rag

rag = auto_rag("docs/")
evaluator = RAGEvaluator(
    query_fn=rag.query,
    llm_complete=rag.ai._call,
)

# Auto-generate test set
test_set = evaluator.generate_test_set(rag._documents, num_questions=20)

# Run evaluation
metrics = evaluator.evaluate(test_set)
# Keys: answer_relevance, answer_relevance_llm, context_precision,
#        faithfulness, faithfulness_llm, answer_similarity

# Per-metric aggregates
print(metrics["faithfulness_llm"]["mean"])
print(metrics["answer_relevance"]["std"])
```

## Auto-RAG Tier 2 Features

### 🧭 Smart Query Routing (Feature 6)

Route queries to specialized RAG pipelines based on intent:

```python
from runeextract import auto_rag

# Create specialized RAGs
rag_tech = auto_rag("./docs/engineering/")
rag_legal = auto_rag("./docs/contracts/")
rag_finance = auto_rag("./docs/reports/")

# Create a routed RAG
rag = auto_rag("./docs/", routing_rags={
    "technical": rag_tech,
    "legal": rag_legal,
    "financial": rag_finance,
})

# Auto-routes to the right RAG based on intent
result = rag.query("What is the API rate limit?")       # → technical
result = rag.query("What are the contract terms?")       # → legal
result = rag.query("What was Q3 revenue?")               # → financial
```

Each query is automatically classified via keyword + historical feedback and routed to the best-fit RAG.

### 🧪 A/B Experiments (Feature 9)

Test multiple RAG configurations with automatic user bucketing:

```python
from runeextract import auto_rag

rag = auto_rag("./docs/", experiment_config={
    "name": "chunking_strategy",
    "variants": {
        "control": {"chunk_size": 1000, "top_k": 5},
        "treatment": {"chunk_size": 500, "top_k": 10, "reranker": "cross-encoder"},
    },
})

# User ID determines variant (consistent bucketing)
result = rag.query("What is the policy?", user_id="user123")

# Feedback collection improves variant selection
result = rag.query("Explain the findings.", user_id="user456")
```

Built-in statistical significance testing, multi-metric optimization, and experiment reports via `rag._experiment_manager.get_report()`.

### 🌐 Multi-Language Support (Feature 10)

Auto-detect and translate queries across languages:

```python
from runeextract import auto_rag

rag = auto_rag("./docs/", multi_language=True)

# Query in any supported language — answer comes back in same language
result = rag.query("¿Cuál es la política de trabajo remoto?")  # Spanish
print(result.answer)  # → Respuesta en español

result = rag.query("Wie hoch war der Umsatz im dritten Quartal?")  # German
print(result.answer)  # → Antwort auf Deutsch
```

Language detection, cross-lingual embeddings, and translation caching happen automatically.

### 🧠 Chain-of-Thought Reasoning (Feature 14)

Multi-step reasoning for complex, analytical queries:

```python
from runeextract import auto_rag

rag = auto_rag("./docs/", reasoning=True)

# Complex query gets decomposed into sub-questions
result = rag.query(
    "Compare the revenue growth in Q1 vs Q2 and explain "
    "the factors that caused the difference."
)
# Internally decomposes into:
#   1. What was Q1 revenue?
#   2. What was Q2 revenue?
#   3. What factors affected the difference?

# View reasoning trace
trace = result.reasoning_trace
for step in trace.steps:
    print(f"Step {step.step_number}: {step.question}")
    print(f"  Answer: {step.answer[:100]}")

print(f"Final confidence: {trace.final_confidence:.2%}")
```

Per-query override: `rag.query("Complex question?", reasoning=True)`.

### 🔐 RBAC with User Permissions (Feature 11)

Role-based access control filters chunks per user:

```python
from runeextract import auto_rag

rag = auto_rag("./docs/", rbac=True)

# Configure permissions
rag._rbac.set_permissions({
    "finance/*": ["finance_team", "executives"],
    "hr/*": ["hr_team"],
    "public/*": ["*"],
})

# Query with user context — only permitted chunks are returned
result = rag.query("What is the bonus policy?",
                   user="alice@company.com",
                   roles=["engineer"])
```

Field-level redaction and full audit logging included.

### ⚡ Streaming RAG (Feature 12)

Progressive refinement with token-level streaming:

```python
from runeextract import auto_rag
from runeextract.rag import StreamEventType

rag = auto_rag("./docs/", streaming=True)

for event in rag.query_stream("What are the key findings?"):
    if event.type == StreamEventType.PARTIAL_ANSWER:
        print(event.text, end="", flush=True)
    elif event.type == StreamEventType.COMPLETE:
        print(f"\nConfidence: {event.confidence:.2%}")
```

Stages: retrieval → partial answer → refinement → citations → complete.

### 📊 Analytics Dashboard (Feature 8)

Query metrics, time series, latency percentiles, CSV/JSON export:

```python
from runeextract import auto_rag

rag = auto_rag("./docs/", analytics=True)
result = rag.query("What is the revenue?")

# Get analytics
analytics = rag.get_analytics()
summary = analytics.get_summary()
print(summary.to_dict())

# Export
analytics.export_csv("rag_metrics.csv")
```

### 🧠 Semantic Caching (Feature 7)

Embedding-based cache avoids re-querying semantically identical questions:

```python
from runeextract import auto_rag

rag = auto_rag("./docs/", semantic_cache=True)
result = rag.query("What is the revenue?")  # Uncached — runs full pipeline
result = rag.query("How much revenue?")     # Cache hit — returns instantly
```

Configurable similarity threshold and TTL via `cache_similarity` and `cache_ttl`.

### 🚀 RAG-as-a-Service API (Feature 13)

Expose your RAG pipeline as a FastAPI REST API:

```python
from runeextract import auto_rag

rag = auto_rag("./docs/")

# Create the API server
api = rag.create_api_server(api_keys=["my-secret-key"], rate_limit=100)

# Get the FastAPI app
app = api.create_app()

# Run with: uvicorn app:app --reload

# Or start directly
rag.serve(host="0.0.0.0", port=8000, api_keys=["my-secret-key"])
```

Endpoints: `/health`, `/query`, `/query/stream`, `/ingest`, `/metrics`.
```
