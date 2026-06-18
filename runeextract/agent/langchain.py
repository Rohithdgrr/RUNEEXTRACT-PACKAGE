"""LangChain integration — RuneExtractLoader for LangChain document loading.

Usage:
    from runeextract.agent import RuneExtractLoader

    loader = RuneExtractLoader("document.pdf")
    docs = loader.load()
"""

from typing import Iterator, List, Optional

from runeextract import extract


class RuneExtractLoader:
    """LangChain-compatible document loader using RuneExtract.

    Mimics the LangChain BaseLoader interface (load/lazy_load).
    Each Document's page_content is the extracted text, with metadata
    copied from the extraction result.
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
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator:
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
        yield {"page_content": doc.text or "", "metadata": meta}

    @classmethod
    def from_file_list(cls, file_paths: List[str], **kwargs) -> List["RuneExtractLoader"]:
        return [cls(fp, **kwargs) for fp in file_paths]
