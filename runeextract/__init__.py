"""
RuneExtract - One extraction API for every document type.
"""

from typing import Optional, List
from runeextract.core.router import ExtractorRouter
from runeextract.models.document import Document, ChunkingStrategy

__version__ = "0.1.0"
__all__ = ["extract", "extract_many", "Document", "ChunkingStrategy"]


def extract(
    file_path: str,
    ocr: bool = False,
    tables: bool = True,
    images: bool = True,
    metadata: bool = True,
    chunking_strategy: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    **kwargs
) -> Document:
    """
    Extract content from a document file.
    
    This is the main entry point for RuneExtract. It automatically detects
    the file type and uses the appropriate extractor.
    
    Args:
        file_path: Path to the file to extract (supports PDF, DOCX, HTML, Markdown, etc.)
        ocr: Enable OCR for images and scanned documents
        tables: Extract tables from the document
        images: Extract images from the document
        metadata: Extract document metadata
        chunking_strategy: Strategy for chunking text (by_page, by_heading, semantic, fixed_size)
        chunk_size: Target chunk size in characters
        chunk_overlap: Character overlap between chunks
        **kwargs: Additional extractor-specific options
        
    Returns:
        Document object with extracted content
        
    Example:
        >>> from runeextract import extract
        >>> doc = extract("report.pdf")
        >>> print(doc.text)
        >>> print(doc.tables)
        >>> print(doc.chunks())
    """
    options = {
        'ocr': ocr,
        'tables': tables,
        'images': images,
        'metadata': metadata,
        'chunking_strategy': chunking_strategy,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        **kwargs
    }
    
    # Get the appropriate extractor
    extractor = ExtractorRouter.get_extractor(file_path, **options)
    
    # Extract content
    document = extractor.extract(file_path)
    
    # Apply chunking if requested
    if chunking_strategy:
        strategy = ChunkingStrategy(chunking_strategy)
        document.chunks(strategy=strategy, size=chunk_size, overlap=chunk_overlap)
    
    return document


def extract_many(
    file_paths: List[str],
    **kwargs
) -> List[Document]:
    """
    Extract content from multiple files.
    
    Args:
        file_paths: List of file paths to extract
        **kwargs: Options passed to extract()
        
    Returns:
        List of Document objects
        
    Example:
        >>> from runeextract import extract_many
        >>> docs = extract_many(["a.pdf", "b.docx", "c.html"])
    """
    documents = []
    for file_path in file_paths:
        try:
            doc = extract(file_path, **kwargs)
            documents.append(doc)
        except Exception as e:
            # Log error but continue with other files
            print(f"Error extracting {file_path}: {e}")
    
    return documents
