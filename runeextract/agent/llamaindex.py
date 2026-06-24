"""LlamaIndex integration — RuneExtractReader for LlamaIndex document loading.

When ``llama-index-core`` is installed, ``load_data()`` returns proper
``llama_index.core.Document`` objects.  Without it, plain dicts are
returned so the integration never requires a hard dependency.

Usage::

    from runeextract.agent import RuneExtractReader

    reader = RuneExtractReader()
    docs = reader.load_data("document.pdf")  # list of Document objects (or dicts)
"""

from typing import Any, List, Optional

from runeextract import extract


def _build_docs(doc) -> List[Any]:
    """Convert a RuneExtract Document to native LlamaIndex Documents."""
    meta = {"source": doc.source_path, "source_type": doc.source_type}
    if doc.metadata:
        meta.update(doc.metadata)

    try:
        from llama_index.core import Document as LlamaindexDocument

        if doc._chunks:
            return [
                LlamaindexDocument(text=c.text, extra_info={**meta, "chunk_id": c.chunk_id})
                for c in doc._chunks
            ]
        return [LlamaindexDocument(text=doc.text or "", extra_info=meta)]
    except ImportError:
        if doc._chunks:
            return [{"text": c.text, "metadata": {**meta, "chunk_id": c.chunk_id}} for c in doc._chunks]
        return [{"text": doc.text or "", "metadata": meta}]


class RuneExtractReader:
    """Read documents via RuneExtract, yielding LlamaIndex-compatible objects.

    Args:
        ocr: Enable OCR for scanned documents/images.
        tables: Extract tables (default True).
        images: Extract images (default True).
        metadata: Extract metadata (default True).
        chunk: If True (default), chunk the document and return one LlamaIndex
            Document per chunk.  If False, return the full text as a single
            Document.
        chunk_strategy: Chunking strategy name (default ``"fixed_size"``).
        chunk_size: Target chunk size (default 1000).
        chunk_overlap: Overlap between chunks (default 100).
        **kwargs: Passed through to ``runeextract.extract()``.
    """

    def __init__(
        self,
        ocr: bool = False,
        tables: bool = True,
        images: bool = True,
        metadata: bool = True,
        chunk: bool = True,
        chunk_strategy: str = "fixed_size",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        **kwargs,
    ):
        self.ocr = ocr
        self.tables = tables
        self.images = images
        self.metadata = metadata
        self.chunk = chunk
        self.chunk_strategy = chunk_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.kwargs = kwargs

    def load_data(self, file_path: str) -> List[Any]:
        """Extract a file and return Documents (native objects or dict fallback).

        Args:
            file_path: Path to the document file.

        Returns:
            List of ``llama_index.core.Document`` objects (or plain dicts
            when ``llama-index-core`` is not installed).
        """
        doc = extract(
            file_path,
            ocr=self.ocr,
            tables=self.tables,
            images=self.images,
            metadata=self.metadata,
            **self.kwargs,
        )
        if self.chunk:
            doc.chunks(strategy=self.chunk_strategy, size=self.chunk_size, overlap=self.chunk_overlap)
        return _build_docs(doc)
