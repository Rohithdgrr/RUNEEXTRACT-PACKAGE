# RuneExtract

**One extraction API for every document type.**

RuneExtract is a universal document extraction library that provides a single, consistent API for extracting content from any file type. Whether it's PDF, DOCX, HTML, images, YouTube videos, or Notion exports - RuneExtract returns the same structured output every time.

## Vision

```python
from runeextract import extract

data = extract("report.pdf")

print(data.text)
print(data.tables)
print(data.images)
print(data.metadata)
print(data.chunks())
```

One API. Any file. Same output schema.

## Installation

```bash
pip install runeextract
```

For OCR support:
```bash
pip install runeextract[ocr]
```

For AI features:
```bash
pip install runeextract[ai]
```

## Quick Start

### Basic Usage

```python
from runeextract import extract

# Extract from any file type
doc = extract("report.pdf")

# Access extracted content
print(doc.text)           # Full text content
print(doc.tables)         # List of tables
print(doc.images)         # List of images
print(doc.metadata)       # Document metadata
print(doc.chunks())       # Chunked content for RAG
```

### With Options

```python
doc = extract(
    "report.pdf",
    ocr=True,
    tables=True,
    chunking_strategy="semantic"
)
```

### Batch Processing

```python
from runeextract import extract_many

docs = extract_many([
    "a.pdf",
    "b.docx",
    "c.html"
])
```

## Universal Schema

All extractors return the same `Document` schema:

```python
class Document:
    text: str                    # Full text content
    tables: List[Table]          # Extracted tables
    images: List[Image]          # Extracted images
    metadata: dict               # Document metadata
    chunks: List[Chunk]          # Chunked content
    source_type: str             # File type identifier
```

## Supported File Types

| Format | Status | Extracted Content |
|--------|--------|-------------------|
| PDF | ✅ MVP | text, tables, images, metadata |
| DOCX | ✅ MVP | paragraphs, tables, images, headers, footers |
| PPTX | ✅ MVP | slides, speaker notes, images |
| XLSX | ✅ MVP | worksheets, tables, formulas |
| HTML | ✅ MVP | headings, paragraphs, tables, links |
| Markdown | ✅ MVP | headings, lists, code blocks, tables |
| Images | ✅ v0.2 | text (OCR), bounding boxes, confidence |
| Scanned PDFs | ✅ v0.2 | text via OCR |
| YouTube | ✅ v0.3 | transcript, timestamps, chapters, metadata |
| Notion | ✅ v0.3 | pages, databases, content |

## Features

### Phase 1: Core Extractors (MVP)

- **PDF**: Extract text, tables, images, and metadata using PyMuPDF and pdfplumber
- **DOCX**: Extract paragraphs, tables, images, headers, and footers
- **PPTX**: Extract slides, text, tables, and images using python-pptx
- **XLSX**: Extract worksheets, tables, and metadata using openpyxl
- **HTML**: Parse headings, paragraphs, tables, and links with BeautifulSoup
- **Markdown**: Extract headings, lists, code blocks, and tables

### Phase 2: OCR Support

Extract text from images and scanned documents:

```python
doc = extract("invoice.jpg", ocr=True)
# Returns: text, bounding boxes, confidence scores
```

Supports:
- Images (JPG, PNG, etc.)
- Scanned PDFs (automatic detection and OCR processing)

### Phase 3: Advanced Table Extraction

Unified table extraction across formats:

```python
class Table:
    rows: List[List[str]]
    columns: List[str]
    dataframe: pd.DataFrame
```

Supported for: PDF, DOCX, HTML, XLSX

### Phase 4: Intelligent Chunking

Optimize content for RAG applications:

```python
chunks = doc.chunks(
    strategy="semantic",  # by_page, by_heading, semantic, fixed_size
    size=1000
)
```

Chunking strategies:
- `by_page`: Split by document pages
- `by_heading`: Split by document structure
- `semantic`: AI-powered semantic chunking
- `fixed_size`: Fixed-length chunks

### Phase 5: Automatic Metadata

Extract rich metadata automatically:

```python
{
    "title": "",
    "author": "",
    "created_at": "",
    "language": "",
    "keywords": []
}
```

### Phase 6: YouTube Integration

Extract video content:

```python
doc = extract("https://youtube.com/watch?v=...")
# Returns: transcript, timestamps, chapters, metadata
```

### Phase 7: Notion Import

Import Notion exports:

```python
doc = extract("notion_export.zip")
# Returns: pages, databases, content
```

### Phase 8: CLI Tool

Command-line interface for quick extraction:

```bash
# Basic extraction
runeextract file.pdf

# Advanced options
runeextract file.pdf --chunks --ocr --tables --output document.json
```

### Phase 9: Async Processing

For large files and batch processing:

```python
from runeextract import extract_async

doc = await extract_async("large.pdf")
```

### Phase 10: AI Features (Optional)

Enhanced analysis with AI:

```python
pip install runeextract[ai]

doc = extract("report.pdf")
print(doc.summary())
print(doc.keywords())
print(doc.entities())
print(doc.questions())
print(doc.flashcards())
```

## Plugin System

Extend RuneExtract with custom extractors:

```python
from runeextract.core.registry import register_extractor

@register_extractor(".epub")
class EPUBExtractor:
    def extract(self, file_path):
        # Your extraction logic
        return Document(...)
```

Then use it seamlessly:

```python
extract("book.epub")  # Works automatically
```

## Project Structure

```
runeextract/
├── core/
│   ├── extractor.py      # Base extractor class
│   ├── registry.py       # Plugin registry
│   ├── router.py         # File type routing
│   └── schemas.py        # Data models
│
├── extractors/
│   ├── pdf/              # PDF extraction
│   ├── docx/             # DOCX extraction
│   ├── pptx/             # PPTX extraction
│   ├── xlsx/             # XLSX extraction
│   ├── html/             # HTML extraction
│   ├── markdown/         # Markdown extraction
│   ├── image/            # Image/OCR extraction
│   ├── audio/            # Audio extraction
│   ├── video/            # Video extraction
│   ├── youtube/          # YouTube extraction
│   └── notion/           # Notion extraction
│
├── processors/
│   ├── ocr.py            # OCR processing
│   ├── tables.py         # Table extraction
│   ├── chunking.py       # Content chunking
│   ├── metadata.py       # Metadata extraction
│   └── cleaning.py       # Text cleaning
│
├── models/
│   ├── document.py       # Document model
│   ├── table.py          # Table model
│   ├── image.py          # Image model
│   └── chunk.py          # Chunk model
│
├── cli/
│   └── main.py           # CLI interface
│
└── tests/
```

## Architecture

```
File
 ↓
Router (detects file type)
 ↓
Appropriate Extractor
 ↓
Normalization Layer
 ↓
Document Object (unified schema)
 ↓
Return
```

## Dependencies

### Core
- `pymupdf` - PDF processing
- `pdfplumber` - Advanced PDF table extraction
- `python-docx` - DOCX processing
- `python-pptx` - PPTX processing
- `openpyxl` - XLSX processing
- `pandas` - Data manipulation
- `beautifulsoup4` - HTML parsing
- `lxml` - XML/HTML parsing
- `markdown-it-py` - Markdown parsing

### OCR (optional)
- `easyocr` or `rapidocr` - Text recognition

### YouTube (optional)
- `youtube-transcript-api` - Transcript extraction
- `yt-dlp` - Video metadata

### AI Features (optional)
- `openai` or similar - AI-powered analysis

## Development Roadmap

### v0.1 (MVP) — ✅ Current Release
- ✅ PDF extraction
- ✅ DOCX extraction
- ✅ PPTX extraction
- ✅ XLSX extraction
- ✅ HTML extraction
- ✅ Markdown extraction
- ✅ CLI interface
- ✅ Chunking strategies
- ✅ Plugin system

### v0.2 (Planned)
- ⏳ OCR support (images and scanned PDFs)
- ⏳ YouTube integration
- ⏳ Notion import
- ⏳ Async processing
- ⏳ AI features

## Contributing

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Rohithdgrr/RUNEEXTRACT-PACKAGE.git
cd runeextract

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev,ocr,ai]"

# Run tests
pytest

# Run linting
black runeextract/
flake8 runeextract/
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Why RuneExtract?

The current ecosystem requires different libraries for different file types:

```python
PyPDF           → PDF
python-docx     → DOCX
BeautifulSoup   → HTML
EasyOCR         → Images
```

RuneExtract unifies all of this:

```python
extract(anything)
```

That simplicity is the entire product.

## Acknowledgments

Built with inspiration from the document processing community and the need for a unified extraction API.

## Contact

- GitHub: [Rohithdgrr/RUNEEXTRACT-PACKAGE](https://github.com/Rohithdgrr/RUNEEXTRACT-PACKAGE)
- Issues: [GitHub Issues](https://github.com/Rohithdgrr/RUNEEXTRACT-PACKAGE/issues)
- Discussions: [GitHub Discussions](https://github.com/Rohithdgrr/RUNEEXTRACT-PACKAGE/discussions)

---

**RuneExtract - One API to extract them all.**
