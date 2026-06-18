"""LlamaIndex integration — RuneExtractReader for LlamaIndex document loading.

Usage:
    from runeextract.agent import RuneExtractReader

    reader = RuneExtractReader()
    docs = reader.load_data("document.pdf")
"""

from typing import List, Optional

from runeextract import extract


class RuneExtractReader:
    """LlamaIndex-compatible document reader using RuneExtract.

    Mimics the LlamaIndex BaseReader interface (load_data).
    Returns dicts with 'text' and 'metadata' keys.
    """

    def __init__(
        self,
        ocr: bool = False,
        tables: bool = True,
        images: bool = True,
        metadata: bool = True,
        **kwargs,
    ):
        self.ocr = ocr
        self.tables = tables
        self.images = images
        self.metadata = metadata
        self.kwargs = kwargs

    def load_data(self, file_path: str) -> List[dict]:
        doc = extract(
            file_path,
            ocr=self.ocr,
            tables=self.tables,
            images=self.images,
            metadata=self.metadata,
            **self.kwargs,
        )
        meta = {"source": doc.source_path, "source_type": doc.source_type}
        if doc.metadata:
            meta.update(doc.metadata)
        return [{"text": doc.text or "", "metadata": meta}]
