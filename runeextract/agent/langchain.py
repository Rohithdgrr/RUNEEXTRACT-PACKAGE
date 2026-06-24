"""LangChain integration — RuneExtractLoader for LangChain document loading.

When ``langchain-core`` is installed, ``load()`` returns proper
``langchain_core.documents.Document`` objects.  Without it, plain dicts
are returned so the integration never requires a hard dependency.

Usage::

    from runeextract.agent import RuneExtractLoader

    loader = RuneExtractLoader("document.pdf")
    docs = loader.load()  # list of Document objects (or dicts)

    from runeextract.agent import RuneExtractTransformer

    transformer = RuneExtractTransformer(chunk_strategy="semantic")
    chunked = transformer.transform_documents(docs)
"""

from typing import Any, Iterator, List, Optional

from runeextract import extract


def _build_doc(text: str, meta: dict) -> Any:
    """Return a native LangChain Document or a plain dict as fallback."""
    try:
        from langchain_core.documents import Document
        return Document(page_content=text, metadata=meta)
    except ImportError:
        return {"page_content": text or "", "metadata": meta}


class RuneExtractLoader:
    """Load documents via RuneExtract, yielding LangChain-compatible objects.

    Args:
        file_path: Path to the document file.
        ocr: Enable OCR for scanned documents/images.
        tables: Extract tables (default True).
        images: Extract images (default True).
        metadata: Extract metadata (default True).
        **kwargs: Passed through to ``runeextract.extract()``.
    """

    def __init__(
        self,
        file_path: str,
        ocr: bool = False,
        tables: bool = True,
        images: bool = True,
        metadata: bool = True,
        **kwargs,
    ):
        self.file_path = file_path
        self.ocr = ocr
        self.tables = tables
        self.images = images
        self.metadata = metadata
        self.kwargs = kwargs

    def load(self) -> list:
        """Extract and return a list of Document objects (one per file)."""
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator:
        """Extract and yield Document objects lazily."""
        doc = extract(
            self.file_path,
            ocr=self.ocr,
            tables=self.tables,
            images=self.images,
            metadata=self.metadata,
            **self.kwargs,
        )
        meta = {"source": doc.source_path, "source_type": doc.source_type}
        if doc.metadata:
            meta.update(doc.metadata)
        yield _build_doc(doc.text or "", meta)

    @classmethod
    def from_file_list(cls, file_paths: List[str], **kwargs) -> List["RuneExtractLoader"]:
        """Create a loader per file path for batch loading."""
        return [cls(fp, **kwargs) for fp in file_paths]


class RuneExtractTransformer:
    """Transform LangChain documents using RuneExtract chunking.

    Implements LangChain's ``BaseDocumentTransformer`` interface.
    """

    def __init__(
        self,
        chunk_strategy: str = "fixed_size",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ):
        self.chunk_strategy = chunk_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def transform_documents(
        self, documents: List[Any], **kwargs
    ) -> List[Any]:
        """Split each document into smaller chunks.

        Args:
            documents: List of LangChain Document objects (or dicts with
                ``page_content``).

        Returns:
            Flat list of chunked Document objects.
        """
        from runeextract.models.document import Document as RuneDocument

        result: list = []
        for doc in documents:
            if isinstance(doc, dict):
                text = doc.get("page_content", doc.get("text", ""))
                meta = doc.get("metadata", {})
            else:
                text = getattr(doc, "page_content", str(doc))
                meta = getattr(doc, "metadata", {})

            rd = RuneDocument(text=text, metadata=meta)
            rd.chunks(strategy=self.chunk_strategy, size=self.chunk_size, overlap=self.chunk_overlap)

            for chunk in rd._chunks:
                chunk_meta = {
                    **meta,
                    "chunk_id": chunk.chunk_id,
                    "source": meta.get("source", ""),
                }
                result.append(_build_doc(chunk.text, chunk_meta))
        return result
