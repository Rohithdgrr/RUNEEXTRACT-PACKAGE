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
