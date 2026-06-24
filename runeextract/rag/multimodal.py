"""Multi-modal RAG — index and retrieve text, tables, and images together."""

import base64
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from runeextract.models.document import Document, Image, Table
from runeextract.rag.types import ChunkWithScore
from runeextract.utils.maturity import experimental, beta

logger = logging.getLogger(__name__)


@dataclass
class MultiModalItem:
    """A single indexed item with type discrimination."""
    text: str
    item_type: str  # "text", "table", "image"
    score: float = 0.0
    source: str = ""
    page: Optional[int] = None
    image_data: Optional[str] = None  # base64-encoded image bytes
    image_format: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiModalResult:
    """Result of a multi-modal search."""
    items: List[MultiModalItem] = field(default_factory=list)
    query: str = ""
    latency_ms: float = 0.0

    @property
    def texts(self) -> List[MultiModalItem]:
        return [i for i in self.items if i.item_type == "text"]

    @property
    def tables(self) -> List[MultiModalItem]:
        return [i for i in self.items if i.item_type == "table"]

    @property
    def images(self) -> List[MultiModalItem]:
        return [i for i in self.items if i.item_type == "image"]

    def to_openai_messages(
        self,
        system_prompt: str = "",
        question: str = "",
        max_images: int = 4,
    ) -> List[Dict[str, Any]]:
        """Format retrieved items as OpenAI multi-modal messages.

        Text and table items are concatenated as context. Image items are
        included as image_url content blocks (up to ``max_images``).
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        context_parts = []
        image_blocks = []

        for item in self.items:
            if item.item_type in ("text", "table"):
                label = item.item_type.upper()
                context_parts.append(f"[{label}] {item.text}")
            elif item.item_type == "image" and len(image_blocks) < max_images:
                if item.image_data:
                    image_blocks.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{item.image_format};base64,{item.image_data}",
                        },
                    })
                    if item.text:
                        context_parts.append(f"[IMAGE] {item.text}")

        user_content: List[Dict[str, Any]] = []

        if context_parts:
            user_content.append({
                "type": "text",
                "text": "Context:\n" + "\n".join(context_parts),
            })

        user_content.extend(image_blocks)

        if question:
            user_content.append({"type": "text", "text": f"\nQuestion: {question}"})

        messages.append({"role": "user", "content": user_content})
        return messages


def _table_to_text(table: Table) -> str:
    """Convert a Table to a CSV-like text representation."""
    parts = []
    if table.columns:
        parts.append(", ".join(table.columns))
    if table.rows:
        parts.extend(", ".join(row) for row in table.rows)
    return "\n".join(parts)


def _image_to_base64(image: Image) -> str:
    """Convert Image bytes to base64 string."""
    return base64.b64encode(image.data).decode("ascii")


@beta(name="rag.multimodal")
class MultiModalIndex:
    """Index text, tables, and images from a Document for multi-modal retrieval.

    Usage::

        from runeextract.rag.multimodal import MultiModalIndex

        index = MultiModalIndex()
        index.add_document(doc)
        results = index.search("What does the chart show?")
        # results.texts, results.tables, results.images
        messages = results.to_openai_messages(question="Describe the chart")
    """

    def __init__(self, embed_fn: Optional[Any] = None):
        self._items: List[MultiModalItem] = []
        self._embed_fn = embed_fn

    def add_document(
        self,
        doc: Document,
        chunk_strategy: str = "fixed_size",
        chunk_size: int = 500,
    ) -> int:
        """Index all content from a Document: text chunks, tables, and images.

        Args:
            doc: The document to index.
            chunk_strategy: Chunking strategy for text.
            chunk_size: Chunk size for text.

        Returns:
            Number of items indexed.
        """
        from runeextract.models.document import ChunkingStrategy
        strategy = ChunkingStrategy(chunk_strategy)
        chunks = doc.chunks(strategy=strategy, size=chunk_size)
        source = doc.source_path or ""

        count = 0

        for chunk in chunks:
            self._items.append(MultiModalItem(
                text=chunk.text,
                item_type="text",
                source=source,
                page=chunk.metadata.get("page"),
                metadata={"chunk_id": chunk.chunk_id, "document_id": doc.document_id},
            ))
            count += 1

        for table in doc.tables:
            table_text = _table_to_text(table)
            self._items.append(MultiModalItem(
                text=table_text,
                item_type="table",
                source=source,
                page=table.page_number,
                metadata={"document_id": doc.document_id},
            ))
            count += 1

        for img in doc.images:
            b64 = _image_to_base64(img)
            caption = img.caption or ""
            self._items.append(MultiModalItem(
                text=caption,
                item_type="image",
                source=source,
                page=img.page_number,
                image_data=b64,
                image_format=img.format,
                metadata={"document_id": doc.document_id},
            ))
            count += 1

        return count

    def search(
        self,
        query: str,
        top_k: int = 10,
        type_filter: Optional[str] = None,
    ) -> MultiModalResult:
        """Search across all indexed items.

        Uses embedding similarity if ``embed_fn`` is available, otherwise
        falls back to word-overlap scoring.

        Args:
            query: The search query.
            top_k: Maximum items to return.
            type_filter: Optional type to filter ("text", "table", "image").

        Returns:
            A ``MultiModalResult`` with scored and filtered items.
        """
        import time
        start = time.perf_counter()

        candidates = self._items
        if type_filter:
            candidates = [i for i in candidates if i.item_type == type_filter]

        if not candidates:
            return MultiModalResult(query=query, latency_ms=0.0)

        if self._embed_fn is not None:
            query_vec = self._embed_fn([query])[0]
            scored = []
            for item in candidates:
                item_vec = self._embed_fn([item.text])[0]
                score = self._cosine_sim(query_vec, item_vec)
                scored.append((item, score))
        else:
            query_words = set(query.lower().split())
            scored = []
            for item in candidates:
                text_words = set(item.text.lower().split())
                if not query_words or not text_words:
                    score = 0.0
                else:
                    intersection = query_words & text_words
                    score = len(intersection) / max(len(query_words), 1)
                scored.append((item, score))

        scored.sort(key=lambda x: -x[1])
        top = [(item, score) for item, score in scored if score > 0][:top_k]

        latency = (time.perf_counter() - start) * 1000
        items = [item for item, score in top]
        for item, score in top:
            item.score = score

        return MultiModalResult(
            items=items,
            query=query,
            latency_ms=latency,
        )

    @staticmethod
    def _cosine_sim(a: List[float], b: List[float]) -> float:
        dot = sum(ai * bi for ai, bi in zip(a, b))
        na = sum(ai * ai for ai in a) ** 0.5
        nb = sum(bi * bi for bi in b) ** 0.5
        return dot / max(na * nb, 1e-12)

    @property
    def item_count(self) -> int:
        return len(self._items)

    def stats(self) -> Dict[str, int]:
        """Return counts per type."""
        counts: Dict[str, int] = {}
        for item in self._items:
            counts[item.item_type] = counts.get(item.item_type, 0) + 1
        return counts
