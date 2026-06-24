# RuneExtract Quick Reference: 5 Game-Changing Features

## 🚀 One-Line Production RAG

```python
from runeextract import auto_rag

rag = auto_rag(
    "./docs/",
    intelligence="adaptive",  # 🎯 Self-tuning
    watch=True,               # ⚡ Live sync
    incremental=True,         # ⚡ Smart indexing
    multimodal=True,          # 👁️  Vision
    safe_mode=True,           # 🛡️  Security
    cost_limit=10.00          # 💰 Budget cap
)

result = rag.query("question?", cite=True)
```

---

## Feature 1: 🎯 Adaptive Intelligence

**Zero-config self-tuning RAG**

```python
# Basic
rag = auto_rag("./docs/", intelligence="adaptive")

# Auto-detects:
# - Query type (factual/analytical/comparison)
# - Document type (code/academic/dense/sparse)
# - Optimal chunking strategy
# - Best retrieval parameters

# Get stats
stats = rag.get_stats()
print(stats['avg_confidence'])  # 0.87
```

**Auto-Chunking:**
- Code → `by_heading`
- Academic → `by_heading` + hierarchical
- Long docs → RAPTOR
- Default → `sentence_window`

---

## Feature 2: ⚡ Live Document Sync

**Incremental indexing + file watcher**

```python
# Enable live sync
rag = auto_rag("./docs/", watch=True, incremental=True)

# Auto-detects:
# - New files → index
# - Modified files → re-index
# - Deleted files → remove
# - Unchanged files → skip

# Control
rag.stop_watch()
rag.start_watch("./new_dir/")

# Check status
print(rag.get_stats()['watching'])  # True
```

**Hash-based deduplication:**
- SHA-256 of file contents
- Only processes changes
- 10x faster re-indexing

---

## Feature 3: 📚 Citation Engine

**Auto-cite every claim with provenance**

```python
# Enable citations (default: True)
result = rag.query("What is the methodology?", cite=True)

print(result.answer)
# "The study used regression analysis [1]..."

for citation in result.citations:
    print(f"[{citation.chunk_index+1}] {citation.source}")
    print(f"  Page: {citation.page}")
    print(f"  Similarity: {citation.similarity_score:.3f}")
    print(f"  Extracted: {citation.extracted_at}")
    print(f"  Location: {citation.bounding_box}")
```

**Provenance tracking:**
- File path + page number
- Extraction timestamp
- Confidence scores
- Bounding boxes
- Retrieval rank

---

## Feature 4: 👁️ Multi-Modal RAG

**Vision understanding for images/charts/tables**

```python
# Enable multi-modal
rag = auto_rag("./docs/", multimodal=True)

# Queries can reference visual content
result = rag.query("What does the revenue chart show?")
result = rag.query("Describe the architecture diagram")

# Automatically:
# - Analyzes images with vision models
# - Interprets charts and graphs
# - Extracts table data
# - Indexes visual descriptions
```

**Vision caching:**
- Hash-based cache
- Reuses descriptions
- Faster processing

---

## Feature 5: 🛡️ Production Safeguards

**Security + cost controls in one switch**

```python
# Safe mode: ALL protections ON
rag = auto_rag(
    "./docs/",
    safe_mode=True,         # Master switch
    cost_limit=10.00,       # $10 hard cap
    scan_secrets=True,      # Auto-redact
    timeout_per_doc=60      # 60s max
)

# Per-query cost tracking
result = rag.query("question")
print(f"Cost: ${result.cost:.4f}")
print(f"Total: ${result.total_session_cost:.2f}")

# System stats
stats = rag.get_stats()
print(f"Cost: ${stats['total_cost']:.2f} / ${stats['cost_limit']:.2f}")
print(f"Remaining: ${stats['cost_remaining']:.2f}")
```

**Built-in protections:**
- ✅ Cost tracking & hard limits
- ✅ Secret scanning (30+ patterns)
- ✅ Prompt injection defense
- ✅ Rate limiting
- ✅ Circuit breakers
- ✅ Timeout protection

---

## Complete Example

```python
from runeextract import auto_rag

# Production-ready RAG (all 5 features)
rag = auto_rag(
    source="./documents/",
    
    # Feature 1: Adaptive
    intelligence="adaptive",
    chunking="auto",
    
    # Feature 2: Live Sync
    watch=True,
    incremental=True,
    
    # Feature 4: Multi-Modal
    multimodal=True,
    
    # Feature 5: Security
    safe_mode=True,
    cost_limit=25.00,
    
    # Standard
    llm="openai:gpt-4o-mini",
    reranker="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Query with citations
result = rag.query(
    "What are the Q4 revenue projections?",
    cite=True,
    answer_length="medium"
)

# Results
print(result.answer)
print(f"Confidence: {result.confidence:.2%}")
print(f"Cost: ${result.cost:.4f}")
print(f"Citations: {len(result.citations)}")

# Stats
stats = rag.get_stats()
print(f"Documents: {stats['total_documents']}")
print(f"Queries: {stats['total_queries']}")
print(f"Budget used: ${stats['total_cost']:.2f} / ${stats['cost_limit']:.2f}")

# Cleanup
rag.stop_watch()
```

---

## API Reference

### AutoRAG Constructor

```python
rag = auto_rag(
    source: str | List[str],
    
    # Standard params
    embedding="openai:text-embedding-3-small",
    vector_store="chromadb",
    collection_name="documents",
    persist_directory="./chroma_db",
    chunking="auto",
    chunk_size=1000,
    chunk_overlap=100,
    reranker=None,
    llm="openai:gpt-4o-mini",
    ai_processor=None,
    
    # 🚀 Game-changing features
    intelligence="adaptive",
    watch=False,
    incremental=True,
    multimodal=False,
    safe_mode=False,
    cost_limit=None,
    scan_secrets=False,
    timeout_per_doc=300,
    
    **extract_options
)
```

### Query Method

```python
result = rag.query(
    question: str,
    top_k=5,
    metadata_filter=None,
    cite=True,                    # Feature 3
    hyde=False,
    multi_query=False,
    answer_length="medium",       # "short", "medium", "long"
    **llm_kwargs
)
```

### Result Object

```python
result.answer               # str
result.citations            # List[Citation]
result.confidence          # float (0-1)
result.retrieved_chunks    # List[ChunkWithScore]
result.latency_ms          # float
result.tokens_used         # dict
result.cost                # float (Feature 5)
result.total_session_cost  # float (Feature 5)
```

### Citation Object

```python
citation.text              # str
citation.source            # str
citation.page              # int | None
citation.chunk_index       # int
citation.relevance_score   # float
# 🚀 Feature 3 enhancements:
citation.bounding_box      # dict | None
citation.extracted_at      # str | None
citation.retrieval_rank    # int | None
citation.similarity_score  # float | None
```

### Stats Method

```python
stats = rag.get_stats()

stats['total_documents']     # int
stats['total_queries']       # int
stats['avg_latency_ms']      # float
stats['avg_confidence']      # float
stats['total_cost']          # float
stats['cost_limit']          # float | None
stats['cost_remaining']      # float | None
stats['intelligence_mode']   # str
stats['multimodal_enabled']  # bool
stats['safe_mode']           # bool
stats['watching']            # bool
```

---

## Time Savings Breakdown

| Feature | Traditional Approach | RuneExtract | Time Saved |
|---------|---------------------|-------------|-----------|
| Parameter tuning | 4-8 hours | 0 hours | **4-8 hours** |
| Re-indexing pipeline | 30 min per update | Automatic | **30 min/update** |
| Citation system | 2-3 days | 1 parameter | **2-3 days** |
| Image extraction | 1-2 weeks | 1 parameter | **1-2 weeks** |
| Security hardening | 2-3 weeks | 1 parameter | **2-3 weeks** |

**Total: 4-6 weeks saved per RAG project**

---

## Quick Tips

1. **Start simple**: `rag = auto_rag("./docs/", safe_mode=True)`
2. **Enable all features**: Add `intelligence="adaptive"`, `watch=True`, `multimodal=True`
3. **Monitor costs**: Check `stats['cost_remaining']` regularly
4. **Use citations**: Always set `cite=True` for production
5. **Watch performance**: Track `stats['avg_confidence']` to ensure quality

---

## Next Steps

- 📖 Read full guide: [GAME_CHANGING_FEATURES.md](GAME_CHANGING_FEATURES.md)
- 🎮 Try demo: [examples/game_changing_features_demo.py](../examples/game_changing_features_demo.py)
- 📚 API docs: [API.md](API.md)
- 💬 Get help: [GitHub Issues](https://github.com/Rohithdgrr/RUNEEXTRACT-PACKAGE/issues)
