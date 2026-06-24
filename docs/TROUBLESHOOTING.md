# Troubleshooting

Common issues and solutions for RuneExtract v0.6.0.

---

## Installation Issues

### ImportError: No module named 'fitz'

**Cause:** PyMuPDF not installed.

**Solution:**
```bash
pip install pymupdf
# or reinstall runeextract which includes it
pip install --force-reinstall runeextract
```

### DependencyMissingError: (ai) openai

**Cause:** AI features require additional optional dependencies.

**Solution:**
```bash
pip install "runeextract[ai]"
```

For other features:
```bash
pip install "runeextract[ocr]"       # OCR
pip install "runeextract[rag]"       # Token counting
pip install "runeextract[ai]"        # OpenAI provider
pip install "runeextract[all]"       # Everything
```

### OPENAI_API_KEY not found

**Cause:** No API key configured.

**Solution:**
```bash
# Set environment variable
export OPENAI_API_KEY="sk-..."
# On Windows (PowerShell):
# $env:OPENAI_API_KEY="sk-..."

# Or pass directly to AIProcessor
from runeextract.processors.ai import AIProcessor
ai = AIProcessor(api_key="sk-...", provider="openai")
```

---

## Extraction Issues

### UnsupportedFormatError: No extractor found for '.xyz'

**Cause:** File format not recognized or supported.

**Solution:**
- Check file extension is correct
- Supported formats: .pdf, .docx, .pptx, .xlsx, .html, .md, .csv, .json, .png, .jpg, .epub
- For plain text, rename to `.md` (markdown)
- Register a custom extractor (see Plugin System docs)

### Empty text in PDF

**Cause:** Scanned PDF (image-based, no embedded text). The `extract()` function only extracts embedded text by default.

**Solution:**
```python
doc = extract("scanned.pdf", ocr=True)
# Requires: pip install "runeextract[ocr]"
```

### CorruptFileError: File is damaged or password-protected

**Cause:** The file is corrupt, encrypted, or password-protected.

**Solution:**
- Try opening the file in its native application to verify it's not corrupt
- For password-protected files:
  ```python
  doc = extract("protected.pdf", password="mypassword")
  # Supported: PDF (PyMuPDF), DOCX, XLSX (msoffcrypto-tool)
  ```
- Handle the `WrongPasswordError` exception:
  ```python
  from runeextract.exceptions import WrongPasswordError
  try:
      doc = extract("protected.pdf", password="wrong")
  except WrongPasswordError:
      print("Incorrect password")
  ```

### FileTooLargeError: File exceeds size limit

**Cause:** File exceeds `max_file_size` (default depends on configuration).

**Solution:**
```bash
# Increase via environment
export RUNEEXTRACT_MAX_FILE_SIZE=500000000  # 500MB

# Or via config
from runeextract import set_config, get_config
cfg = get_config()
cfg.max_file_size = 500000000
set_config(cfg)
```

### Extraction is slow

**Cause:** Large files, OCR enabled, or AI processing.

**Solutions:**
- Disable OCR for text-based PDFs: `extract("file.pdf", ocr=False)`
- Use streaming for large files:
  ```python
  from runeextract import extract_stream
  async for page_doc in extract_stream("large.pdf"):
      print(page_doc.text)
  ```
- Use `extraction_timeout` to limit wait time:
  ```python
  doc = extract("large.pdf", extraction_timeout=60)  # 60 second timeout
  ```
- Disable cache to avoid disk I/O: `extract("file.pdf", use_cache=False)`

---

## AI & Provider Issues

### AIProcessor connection refused / timeout

**Cause:** Network issues, incorrect API endpoint, or provider outage.

**Solutions:**
- Verify API key is set correctly
- Check internet connection
- For local providers (Ollama), ensure the server is running:
  ```bash
  # Start Ollama
  ollama serve
  ```
- Use the circuit breaker to handle transient failures automatically

### OpenRouter / custom base URL not working

**Cause:** The `OPENAI_BASE_URL` env var is not set, or the provider name is incorrect.

**Solution:**
```bash
# Set the base URL
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export OPENAI_API_KEY="sk-or-v1-..."

# Use the correct provider name
```
```python
ai = AIProcessor(provider="openrouter", model="openai/gpt-4o")
```

Supported OpenAI-compatible providers:
- `openai` (default, uses `api.openai.com`)
- `openrouter` (uses `OPENAI_BASE_URL` or `OPENROUTER_BASE_URL`)
- `ollama` (uses `localhost:11434/v1`)
- `azure` (uses `AZURE_OPENAI_ENDPOINT`)
- `groq` (uses `api.groq.com`)
- `together` (uses `api.together.xyz`)
- `deepseek` (uses `api.deepseek.com`)

### AI responses are empty or truncated

**Cause:** Token limit reached or model returned empty response.

**Solutions:**
- Increase `max_tokens`:
  ```python
  doc.ask("Long question?", max_tokens=4096)
  ```
- Check if the model supports the requested feature (e.g., streaming, function calling)
- Verify `response_format` compatibility with the provider

### Streaming not working

**Cause:** Provider does not support streaming, or `_call_stream` not implemented.

**Solution:**
- Check the feature matrix in [ai.md](ai.md) for streaming support
- Providers that don't support streaming fall back to yielding the full response as a single token
- Currently supported: OpenAI, OpenRouter, Azure, Groq, Together, DeepSeek, Mistral

### ChatSession not remembering context

**Cause:** Incorrect usage — each `ask()` call on a new `ChatSession` creates a fresh conversation.

**Solution:**
```python
# Correct: reuse the same chat session
chat = doc.chat()
answer1 = chat.ask("What is section 1?")  # remembers this
answer2 = chat.ask("Can you elaborate?")  # remembers answer1 + question

# Wrong: creating a new session each time
answer1 = doc.chat().ask("What is section 1?")
answer2 = doc.chat().ask("Can you elaborate?")  # fresh session, no context
```

---

## Windows-Specific Issues

### AttributeError: module 'os' has no attribute 'fchmod'

**Issue:** The `os.fchmod` function is Unix-only and not available on Windows.

**Status:** Fixed in v0.6.0 — extraction falls back gracefully on Windows.

**Workaround for older versions:**
```python
# Use extract(file_path) directly instead of extract_from_bytes()
doc = extract("path/to/file.pdf")
```

### Temp file cleanup fails with "file is being used"

**Issue:** Windows locks temporary files, preventing deletion.

**Solution:**
```python
import os
import time
from runeextract import extract

doc = extract("file.pdf")
# Temp files are cleaned up automatically; this warning is benign.
```

### The process cannot access the file

**Issue:** Another process has locked the file.

**Solution:**
- Close the file in any other application
- Wait a moment and retry
- Verify the file is not open in another Python process

---

## RAG & Vector Store Issues

### chromadb.errors.DuplicateIDError

**Cause:** Duplicate chunk IDs when ingesting with `by_heading` strategy (was a bug in older versions).

**Solution:**
- Upgrade to v0.6.0+ (chunk ID collision is fixed)
- Use a different chunking strategy: `fixed_size`, `semantic`, or `by_page`
- Use `upsert` mode instead of `add`:
  ```python
  # Requires manual ChromaDB collection access
  import chromadb
  client = chromadb.PersistentClient(path="./chroma_db")
  collection = client.get_or_create_collection("docs")
  collection.upsert(ids=ids, documents=texts, embeddings=embeddings)
  ```

### No results from search

**Cause:** Chunks not created yet, or no matching content.

**Solutions:**
- Ensure chunks are created before searching:
  ```python
  doc.chunks(strategy="semantic", size=500)  # must be called first
  results = doc.search("query")
  ```
- Check that the query uses relevant terms from the document
- Try different search modes:
  ```python
  results = doc.search("query", mode="sparse")  # BM25 only
  results = doc.search("query", mode="dense")   # embedding only
  results = doc.search("query", mode="hybrid")  # default
  ```

### FAISS index file not found

**Cause:** Index path doesn't exist or was moved.

**Solution:**
```python
# Save with absolute path
import os
doc.to_faiss(index_path=os.path.abspath("./my_index"))

# Load later
import faiss
index = faiss.read_index(os.path.abspath("./my_index"))
```

---

## CLI Issues

### Command not found: runeextract

**Cause:** Package not installed, or not in PATH.

**Solution:**
```bash
pip install runeextract
# Verify installation
python -m runeextract --help
```

### --ask flag not working

**Cause:** AI provider not configured.

**Solution:**
```bash
export OPENAI_API_KEY="sk-..."
runeextract file.pdf --ask "What is this about?"
```

### --ocr flag not working

**Cause:** OCR extra not installed.

**Solution:**
```bash
pip install "runeextract[ocr]"
runeextract scanned.pdf --ocr
```

---

## Performance Issues

### High memory usage

**Solutions:**
- Stream large files instead of loading fully:
  ```python
  async for page_doc in extract_stream("large.pdf"):
      process(page_doc)
  ```
- Disable options you don't need:
  ```python
  doc = extract("file.pdf", tables=False, images=False, metadata=False)
  ```
- Use `MemoryProfiler` to track usage:
  ```python
  from runeextract import MemoryProfiler
  profiler = MemoryProfiler(warn_mb=500)
  with profiler.profile("extract pdf"):
      doc = extract("large.pdf")
  print(profiler.get_peak_mb())  # peak memory in MB
  ```

### Slow import time

**Cause:** Eager loading of optional dependencies.

**Solutions:**
- All dependencies are loaded lazily — import time should be fast
- If import time is slow, check for conflicting packages:
  ```bash
  pip list | grep -E "pymupdf|openai|chromadb|faiss"
  ```
- Use `python -X importtime` to profile:
  ```bash
  python -X importtime -c "from runeextract import extract" 2> import.log
  ```

---

## Error Codes Reference

| Code | Exception | Description |
|------|-----------|-------------|
| E000 | `ExtractionError` | Base extraction error |
| E001 | `UnsupportedFormatError` | Unsupported file format |
| E002 | `CorruptFileError` | Corrupt or unreadable file |
| E003 | `FileTooLargeError` | File exceeds size limit |
| E004 | `DependencyMissingError` | Missing optional dependency |
| E005 | `ExtractionTimeoutError` | Extraction exceeded timeout |
| E006 | `PathTraversalError` | Path traversal attack detected |
| E033 | `ExtractionError` | Embedding provider not supported |
| E107 | `StructuredExtractionError` | Structured extraction failure |
| E108 | `WrongPasswordError` | Incorrect file password |

---

## Still Having Issues?

1. Search existing [GitHub Issues](https://github.com/Rohithdgrr/RUNEEXTRACT-PACKAGE/issues)
2. Open a new issue with:
   - Python version and OS
   - Full error message and traceback
   - Minimal reproducible example
   - File type and version (if applicable)
   - Package version (`python -c "import runeextract; print(runeextract.__version__)"`)
