# Examples

## Basic Extraction

```python
from runeextract import extract

# PDF
doc = extract("report.pdf")
print(doc.text[:500])
print(doc.metadata)

# DOCX
doc = extract("proposal.docx")
print(doc.tables[0].rows)

# HTML
doc = extract("https://example.com")
print(doc.metadata["title"])

# YouTube
doc = extract("https://youtube.com/watch?v=...")
print(doc.text[:500])  # transcript

# Image with OCR
doc = extract("scanned_doc.png", ocr=True)
print(doc.text)
```

## RAG Pipeline

### Chunk and search

```python
from runeextract import extract
from runeextract.models.document import ChunkingStrategy

doc = extract("research_paper.pdf", chunking_strategy="sentence_window", chunk_size=5)

# Search chunks
results = doc.search("machine learning algorithms", mode="hybrid", top_k=5)
for chunk, score in results:
    print(f"[{score:.3f}] {chunk.text[:100]}...")

# With metadata filter
results = doc.search("deep learning", metadata_filter={"page_number": 1})
```

### Vector store indexing

```python
from runeextract import extract_and_index

doc = extract_and_index(
    "knowledge_base.pdf",
    store="chromadb",
    collection_name="my_docs",
    persist_directory="./chroma_db",
    chunking_strategy="semantic",
)
```

### RAG question answering

```python
doc = extract("company_policy.pdf")
answer = doc.ask("What is the vacation policy?")
print(answer)
```

### Contextual compression

```python
doc = extract("long_article.pdf")
compressed = doc.compress("key findings")
print(compressed)
```

## AI Processing

### With different providers

```python
from runeextract.processors.ai import AIProcessor

# OpenAI (default)
ai = AIProcessor(provider="openai")

# Anthropic Claude
ai = AIProcessor(provider="anthropic", model="claude-3-5-sonnet-20241022")

# Google Gemini
ai = AIProcessor(provider="gemini", model="gemini-1.5-pro")

# Ollama (local)
ai = AIProcessor(provider="ollama", model="llama3.2")

# Groq
ai = AIProcessor(provider="groq", model="llama3-70b-8192")

# DeepSeek
ai = AIProcessor(provider="deepseek", model="deepseek-chat")

# Azure OpenAI
ai = AIProcessor(provider="azure", model="gpt-4o")
```

### Document AI features

```python
doc = extract("report.pdf")

print(doc.summary())       # AI-generated summary
print(doc.keywords(10))    # Top keywords
print(doc.entities())      # Named entities
print(doc.questions(5))    # Generated questions
print(doc.flashcards(10))  # Study flashcards
```

### Function calling / structured output

```python
from runeextract.processors.ai import AIProcessor

ai = AIProcessor()

# Extract entities with guaranteed structure
tools = AIProcessor.extract_entities_tools()
result = ai.call_with_tools(
    "Extract all named entities from this text.",
    "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "extract_entities"}},
)
print(result["tool_calls"])

# Generate flashcards
tools = AIProcessor.generate_flashcards_tools()
result = ai.call_with_tools(
    "Create flashcards from this content.",
    "Python is a programming language created by Guido van Rossum.",
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "generate_flashcards"}},
)
```

### Streaming

```python
# Via AIProcessor
ai = AIProcessor()
for chunk in ai._call_stream("You are helpful.", "Tell me about AI safety."):
    print(chunk, end="", flush=True)

# Via Document
doc = extract("report.pdf")
for chunk in doc.ask_stream("What are the key findings?"):
    print(chunk, end="", flush=True)
```

### Multi-Turn Chat

```python
from runeextract import extract
doc = extract("company_policy.pdf")

# Create a chat session
chat = doc.chat(system_prompt="You are a helpful HR assistant.")

# Multi-turn conversation with memory
answer1 = chat.ask("What is the vacation policy?")
print(answer1)

answer2 = chat.ask("How many days do I get?")  # remembers context
print(answer2)

# Standalone chat without document
from runeextract.models.document import ChatSession
chat = ChatSession(system_prompt="You are a coding tutor.")
chat.add_user_message("Explain Python decorators.")
chat.add_assistant_message("Decorators are functions that modify other functions...")
answer = chat.ask("Give me an example.")
print(answer)
```

### Cost tracking

```python
ai = AIProcessor()
ai._call("You are helpful.", "Summarize this document.")
print(f"Total cost: ${ai.total_cost:.6f}")
print(f"Total tokens in: {ai.total_input_tokens}")
print(f"Total tokens out: {ai.total_output_tokens}")
print(f"API calls: {ai.call_count}")
```

### Query expansion

```python
ai = AIProcessor()

queries = ai.expand_query("What is machine learning?", n_queries=3)
for q in queries:
    print(q)

# HyDE: hypothetical document for embedding
hypothetical = ai.hyde("How does transformer architecture work?")
```

### Rate limiter

```python
from runeextract.utils.rate_limiter import RateLimiter

limiter = RateLimiter(requests_per_minute=10)
ai = AIProcessor(rate_limiter=limiter)

# All calls to _call(), call_with_tools(), etc. will be throttled
```

### Batch processing

```python
ai = AIProcessor()

prompts = [
    {"system": "Summarize.", "user": "Text about AI..."},
    {"system": "Summarize.", "user": "Text about physics..."},
    {"system": "Summarize.", "user": "Text about chemistry..."},
]

results = ai.batch_process(prompts, max_concurrency=4)
for r in results:
    print(r)
```

## Multi-modal

```python
doc = extract("presentation.pptx", images=True)

# OpenAI vision-format messages with base64 images
messages = doc.to_openai_messages(include_images=True)
print(messages[0]["content"])  # list of text + image_url blocks
```

## Export Formats

```python
doc = extract("report.docx")

# Dict / JSON
data = doc.to_dict()
json_str = doc.to_json(indent=2)

# Markdown
md = doc.to_markdown()

# LlamaIndex
llama_docs = doc.to_llamaindex_documents()

# ChromaDB
doc.to_chromadb(collection_name="reports")

# FAISS
doc.to_faiss(index_path="./faiss_index")

# OpenAI messages
messages = doc.to_openai_messages(system_message="Analyze this document.")
```

## Quality Scoring

```python
doc = extract("invoice.pdf", ocr=True)
scores = doc.score_quality()
print(f"Text density: {scores['text_density']}/100")
print(f"Readability: {scores['readability']}/100")
print(f"Structure: {scores['structure']}/100")
print(f"Overall quality: {scores['overall']}/100")
```

## Web Crawling

```python
from runeextract import extract_crawl

docs = extract_crawl(
    "https://docs.example.com",
    max_pages=20,
    same_domain=True,
    delay=0.5,
)

for doc in docs:
    print(f"{doc.source_path}: {len(doc.text)} chars")
```

## Custom Extractor

```python
from runeextract.core.extractor import BaseExtractor
from runeextract.models.document import Document

class MyFormatExtractor(BaseExtractor):
    def extract(self, file_path):
        self.validate_file(file_path)
        # custom extraction logic
        return Document(
            text="extracted content",
            tables=[],
            images=[],
            metadata={"format": "myformat"},
            source_type="myformat",
            source_path=file_path,
        )

    def supported_extensions(self):
        return [".myfmt"]

# Register it
from runeextract.core.registry import ExtractorRegistry
ExtractorRegistry.register(".myfmt")(MyFormatExtractor)

# Now it works
doc = extract("file.myfmt")
```
