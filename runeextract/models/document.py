from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple

from runeextract.models.types import (
    Chunk, Table, Image, ChunkingStrategy,
    _estimate_token_count, _get_token_encoding,
)
from runeextract.models.chunking import (
    chunk_fixed_size, chunk_by_page, chunk_by_heading,
    chunk_semantic, chunk_by_token, chunk_sentence_window,
)
from runeextract.models.chat_session import ChatSession as _ChatSession

logger = logging.getLogger(__name__)

_document_shared_ai = None

def _get_document_ai(ai_processor=None):
    global _document_shared_ai
    if ai_processor is not None:
        return ai_processor
    if _document_shared_ai is None:
        from runeextract.processors.ai import AIProcessor
        _document_shared_ai = AIProcessor()
    return _document_shared_ai


class ChatSession(_ChatSession):
    pass


@dataclass
class Document:
    text: str
    tables: List[Table] = field(default_factory=list)
    images: List[Image] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_type: str = ""
    source_path: Optional[str] = None
    document_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    _chunks: Optional[List[Chunk]] = field(default=None, repr=False)
    _chunk_params: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def token_count(self, encoding_name: str = "cl100k_base") -> int:
        enc = _get_token_encoding(encoding_name)
        if enc:
            return len(enc.encode(self.text))
        return _estimate_token_count(self.text)

    def chunks(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE,
        size: int = 1000,
        overlap: int = 100,
        on_chunk: Optional[Callable[[Chunk], None]] = None,
        **kwargs
    ) -> List[Chunk]:
        params = {"strategy": strategy, "size": size, "overlap": overlap, **kwargs}
        if self._chunks is not None and self._chunk_params == params:
            return self._chunks

        self._chunk_params = params

        if strategy == ChunkingStrategy.FIXED_SIZE:
            self._chunks = chunk_fixed_size(self.text, size, overlap)
        elif strategy == ChunkingStrategy.BY_PAGE:
            self._chunks = chunk_by_page(self.text, self.metadata.get("page_breaks"))
        elif strategy == ChunkingStrategy.BY_HEADING:
            self._chunks = chunk_by_heading(self.text)
        elif strategy == ChunkingStrategy.SEMANTIC:
            self._chunks = chunk_semantic(self.text, size)
        elif strategy == ChunkingStrategy.BY_TOKEN:
            self._chunks = chunk_by_token(self.text, size, overlap, **kwargs)
        elif strategy == ChunkingStrategy.SENTENCE_WINDOW:
            self._chunks = chunk_sentence_window(self.text, size, overlap)
        elif strategy == ChunkingStrategy.HIERARCHICAL:
            self._chunks = self._build_hierarchical_chunks(**kwargs)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        for chunk in self._chunks:
            chunk.parent_document_id = self.document_id

        if on_chunk:
            for chunk in self._chunks:
                on_chunk(chunk)

        return self._chunks

    def _build_hierarchical_chunks(
        self,
        child_size: int = 300,
        parent_size: int = 1500,
        parent_overlap: int = 100,
        child_strategy: str = "fixed_size",
        parent_strategy: str = "fixed_size",
        leaf_size: Optional[int] = None,
    ) -> List[Chunk]:
        if leaf_size is not None:
            child_size = leaf_size
        """Build parent-child hierarchical chunks for RAPTOR-style retrieval.

        Creates two levels:
        - Level 1 (parents): larger chunks for full context
        - Level 0 (children): smaller chunks within each parent for precise retrieval

        Each child chunk has ``parent_chunk_id`` set to its parent's ``chunk_id``.
        Both levels are returned in a flat list.

        Args:
            child_size: Character size for child (leaf) chunks.
            parent_size: Character size for parent chunks.
            parent_overlap: Overlap between parent chunks.
            child_strategy: Chunking strategy for children (fixed_size, sentence_window, etc.).
            parent_strategy: Chunking strategy for parents (fixed_size, by_heading, etc.).

        Returns:
            Flat list of Chunk objects at both child and parent levels.
        """
        from copy import deepcopy

        text = self.text
        if not text:
            return []

        parent_chunks = self._chunk_simple(text, parent_strategy, parent_size, parent_overlap, "hierarchical_parent")
        result: List[Chunk] = []
        parent_id_map: Dict[str, int] = {}

        for p_idx, parent_chunk in enumerate(parent_chunks):
            parent_id = f"parent_{p_idx}"
            parent_chunk.chunk_id = parent_id
            parent_chunk.metadata["strategy"] = "hierarchical"
            parent_chunk.metadata["level"] = 1
            parent_chunk.parent_chunk_id = None
            result.append(parent_chunk)

            child_text = parent_chunk.text
            children = self._chunk_simple(child_text, child_strategy, child_size, 0, "hierarchical_child")
            for c_idx, child in enumerate(children):
                child.chunk_id = f"child_{p_idx}_{c_idx}"
                child.metadata["strategy"] = "hierarchical"
                child.metadata["level"] = 0
                child.parent_chunk_id = parent_id
                child.parent_document_id = self.document_id
                result.append(child)

        return result

    @staticmethod
    def _chunk_simple(
        text: str,
        strategy: str,
        size: int,
        overlap: int,
        label: str,
    ) -> List[Chunk]:
        """Simple chunking helper used internally for hierarchical levels."""
        if not text:
            return []
        chunks = []
        start = 0
        chunk_id = 0
        if strategy == "by_heading":
            heading_pattern = re.compile(r'^(#{1,6}\s+.*)$|^(.+)\n[=\-]+\s*$', re.MULTILINE)
            matches = list(heading_pattern.finditer(text))
            prev_end = 0
            for match in matches:
                if match.start() > prev_end:
                    chunk_text = text[prev_end:match.start()].strip()
                    if chunk_text:
                        chunks.append(Chunk(
                            text=chunk_text,
                            chunk_id=f"tmp_{chunk_id}",
                            start_index=prev_end,
                            end_index=match.start(),
                            metadata={"strategy": label},
                        ))
                        chunk_id += 1
                prev_end = match.end()
            if prev_end < len(text):
                remaining = text[prev_end:].strip()
                if remaining:
                    chunks.append(Chunk(
                        text=remaining,
                        chunk_id=f"tmp_{chunk_id}",
                        start_index=prev_end,
                        end_index=len(text),
                        metadata={"strategy": label},
                    ))
            return chunks

        while start < len(text):
            end = min(start + size, len(text))
            chunk_text = text[start:end]
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=f"tmp_{chunk_id}",
                start_index=start,
                end_index=end,
                metadata={"strategy": label},
            ))
            step = max(size - overlap, 1)
            start += step
            chunk_id += 1
        return chunks

    def hierarchical_chunks(
        self,
        child_size: int = 300,
        parent_size: int = 1500,
        parent_overlap: int = 100,
        child_strategy: str = "fixed_size",
        parent_strategy: str = "fixed_size",
    ) -> List[Chunk]:
        """Build parent-child hierarchical chunks explicitly (aliases ``chunks(HIERARCHICAL)``).

        Returns a flat list containing both parent (level 1) and child (level 0)
        chunks. Child chunks have ``parent_chunk_id`` set.

        Useful for RAPTOR-style retrieval where you search children but return
        parent context.

        Args:
            child_size: Character size for child (leaf) chunks (default 300).
            parent_size: Character size for parent chunks (default 1500).
            parent_overlap: Overlap between parent chunks (default 100).
            child_strategy: Chunking strategy for children (default "fixed_size").
            parent_strategy: Chunking strategy for parents (default "fixed_size").

        Returns:
            Flat list of Chunk objects.
        """
        self._chunks = self._build_hierarchical_chunks(
            child_size=child_size,
            parent_size=parent_size,
            parent_overlap=parent_overlap,
            child_strategy=child_strategy,
            parent_strategy=parent_strategy,
        )
        return self._chunks

    @staticmethod
    def _count_syllables(word: str) -> int:
        word = word.lower().strip(".,!?;:\"'()[]")
        if not word:
            return 0
        vowels = "aeiouy"
        count = 0
        prev_vowel = False
        for ch in word:
            is_vowel = ch in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if count == 0:
            count = 1
        return count

    def _clean_str(self, val: Any) -> Any:
        if isinstance(val, str):
            return val.replace("\x00", "")
        if isinstance(val, dict):
            return {k: self._clean_str(v) for k, v in val.items()}
        if isinstance(val, list):
            return [self._clean_str(v) for v in val]
        return val

    def score_quality(self) -> Dict[str, Any]:
        scores = {}
        text_len = len(self.text) if self.text else 0
        if text_len > 0:
            non_ws = sum(1 for c in self.text if not c.isspace())
            scores["text_density"] = min(100, int((non_ws / text_len) * 100))
        else:
            scores["text_density"] = 0

        if text_len > 50:
            sentences = self.text.count(".") + self.text.count("!") + self.text.count("?")
            words = len(self.text.split())
            syllables = sum(self._count_syllables(w) for w in self.text.split()[:1000])
            if words > 0 and sentences > 0:
                avg_syllables = syllables / min(words, 1000)
                avg_words_per_sent = words / sentences
                raw = 206.835 - (1.015 * avg_words_per_sent) - (84.6 * avg_syllables)
                scores["readability"] = max(0, min(100, int(raw)))
            else:
                scores["readability"] = 50
        else:
            scores["readability"] = 50

        struct_score = 0
        if self.tables:
            struct_score += 30
        if self.images:
            struct_score += 30
        if self.metadata and any(v for v in self.metadata.values() if isinstance(v, str)):
            struct_score += 20
        if text_len > 0:
            struct_score += 20
        scores["structure"] = struct_score

        if text_len > 0:
            scores["completeness"] = 100
        else:
            scores["completeness"] = 0

        ocr_conf = self.metadata.get("ocr_confidence")
        if ocr_conf is not None:
            scores["ocr_confidence"] = min(100, int(float(ocr_conf) * 100))
        else:
            scores["ocr_confidence"] = 50

        weights = {"text_density": 0.20, "readability": 0.15, "structure": 0.25,
                   "completeness": 0.25, "ocr_confidence": 0.15}
        overall = sum(scores[k] * weights[k] for k in weights)
        scores["overall"] = int(overall)
        return scores

    @property
    def quality(self) -> Dict[str, Any]:
        return self.score_quality()

    def to_dict(self) -> Dict[str, Any]:
        return self._clean_str({
            "text": self.text,
            "tables": [
                {
                    "rows": table.rows,
                    "columns": table.columns,
                    "page_number": table.page_number,
                    "caption": table.caption,
                    "metadata": table.metadata
                }
                for table in self.tables
            ],
            "images": [
                {
                    "format": img.format,
                    "width": img.width,
                    "height": img.height,
                    "page_number": img.page_number,
                    "caption": img.caption,
                    "metadata": img.metadata,
                    "data_size": len(img.data)
                }
                for img in self.images
            ],
            "metadata": self.metadata,
            "source_type": self.source_type,
            "source_path": self.source_path,
            "chunks": [
                {
                    "text": chunk.text,
                    "chunk_id": chunk.chunk_id,
                    "start_index": chunk.start_index,
                    "end_index": chunk.end_index,
                    "token_count": chunk.token_count(),
                    "parent_chunk_id": chunk.parent_chunk_id,
                    "metadata": chunk.metadata
                }
                for chunk in (self._chunks or [])
            ]
        })

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines = []
        if self.metadata.get("title"):
            lines.append(f"# {self.metadata['title']}\n")
        lines.append(self.text)
        for i, table in enumerate(self.tables):
            lines.append(f"\n## Table {i + 1}\n")
            sep = "| " + " | ".join(table.columns) + " |"
            lines.append(sep)
            lines.append("| " + " | ".join(["---"] * len(table.columns)) + " |")
            for row in table.rows:
                padded = row + [""] * (len(table.columns) - len(row))
                lines.append("| " + " | ".join(padded) + " |")
        return "\n".join(lines)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def text_length(self) -> int:
        return len(self.text)

    @classmethod
    def merge(cls, documents: List[Document]) -> Document:
        if not documents:
            return cls(text="", source_type="merged")
        merged_text = "\n\n".join(d.text for d in documents)
        merged_tables = []
        merged_images = []
        merged_metadata = {}
        for d in documents:
            merged_tables.extend(d.tables)
            merged_images.extend(d.images)
        return cls(
            text=merged_text,
            tables=merged_tables,
            images=merged_images,
            metadata=merged_metadata,
            source_type="merged",
            source_path=";".join(d.source_path for d in documents if d.source_path),
        )

    def to_openai_messages(self, system_message: Optional[str] = None, include_all: bool = True,
                           include_images: bool = False) -> List[Dict[str, Any]]:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})

        if include_all and self._chunks:
            for chunk in self._chunks:
                messages.append({"role": "user", "content": chunk.text})
        else:
            has_images = include_images and any(hasattr(img, 'data') for img in self.images)
            if has_images:
                content_parts = []
                content_parts.append({"type": "text", "text": self.text})
                for img in self.images:
                    try:
                        import base64
                        b64 = base64.b64encode(img.data).decode("utf-8")
                        mime = f"image/{img.format}" if img.format else "image/png"
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "auto"}
                        })
                    except Exception as exc:
                        logger.warning("Image-to-base64 conversion failed: %s", exc)
                        content_parts.append({"type": "text", "text": f"[IMAGE: {img.format}]"})
                content = content_parts
            else:
                content = self.text
            messages.append({"role": "user", "content": content})
        return messages

    def to_chromadb(self, collection_name: str = "documents", persist_directory: str = "./chroma_db", embedding_function=None,
                    on_insert: Optional[Callable[[str], None]] = None, upsert: bool = False):
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "chromadb is required. Install with: pip install chromadb"
            )
        if self._chunks is None:
            self.chunks()
        client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
        )
        ids = [f"{self.document_id}_{chunk.chunk_id}" for chunk in self._chunks]
        texts = [chunk.text for chunk in self._chunks]
        metadatas = [{
            "source": self.source_path or "",
            "source_type": self.source_type,
            "document_id": self.document_id,
            "chunk_id": chunk.chunk_id,
            "start_index": chunk.start_index,
            "end_index": chunk.end_index,
            **chunk.metadata,
        } for chunk in self._chunks]
        if upsert:
            collection.upsert(documents=texts, ids=ids, metadatas=metadatas)
        else:
            collection.add(documents=texts, ids=ids, metadatas=metadatas)
        if on_insert:
            for cid in ids:
                on_insert(cid)
        return collection

    def to_faiss(self, index_path: str = "./faiss_index", embedding_dim: int = 384):
        try:
            import faiss
            import numpy as np
        except ImportError:
            raise ImportError(
                "faiss and numpy are required. Install with: pip install faiss-cpu numpy"
            )
        import json
        import os

        if self._chunks is None:
            self.chunks()

        rng = np.random.default_rng(42)
        embeddings = rng.random((len(self._chunks), embedding_dim), dtype=np.float32)

        index = faiss.IndexFlatL2(embedding_dim)
        index.add(embeddings)

        metadata_list = [{
            "text": chunk.text,
            "source": self.source_path or "",
            "source_type": self.source_type,
            "document_id": self.document_id,
            "chunk_id": chunk.chunk_id,
            "start_index": chunk.start_index,
            "end_index": chunk.end_index,
            **chunk.metadata,
        } for chunk in self._chunks]

        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
        faiss.write_index(index, index_path + ".index")
        with open(index_path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, ensure_ascii=False, default=str)

        return index, metadata_list

    def _get_ai(self, ai_processor=None):
        return _get_document_ai(ai_processor)

    def search(self, query: str, top_k: int = 5, mode: str = "hybrid",
               metadata_filter: Optional[Dict[str, Any]] = None,
               ai_processor=None) -> List[Tuple[Chunk, float]]:
        if not self._chunks:
            self.chunks()
        if not self._chunks:
            return []

        chunks = self._chunks
        if metadata_filter:
            filtered = []
            for c in chunks:
                matches = all(c.metadata.get(k) == v for k, v in metadata_filter.items())
                if matches:
                    filtered.append(c)
            chunks = filtered
            if not chunks:
                return []

        scores = {}

        if mode in ("dense", "hybrid"):
            ai = self._get_ai(ai_processor)
            query_embedding = ai.embed(query)[0]
            import numpy as np
            chunk_texts = [c.text for c in chunks]
            chunk_embeddings = ai.embed(chunk_texts)
            query_vec = np.array(query_embedding, dtype=np.float32)
            chunk_vecs = np.array(chunk_embeddings, dtype=np.float32)
            norms = np.linalg.norm(chunk_vecs, axis=1) * np.linalg.norm(query_vec)
            dense_scores = np.dot(chunk_vecs, query_vec) / np.maximum(norms, 1e-10)
            for i, s in enumerate(dense_scores):
                scores[id(chunks[i])] = s

        if mode in ("sparse", "hybrid"):
            try:
                from rank_bm25 import BM25Okapi
                tokenized_corpus = [c.text.lower().split() for c in chunks]
                bm25 = BM25Okapi(tokenized_corpus)
                tokenized_query = query.lower().split()
                bm25_scores = bm25.get_scores(tokenized_query)
                import numpy as np
                if bm25_scores.max() > 0:
                    bm25_scores = bm25_scores / bm25_scores.max()
                for i, s in enumerate(bm25_scores):
                    prev = scores.get(id(chunks[i]), 0.0)
                    scores[id(chunks[i])] = prev + s * 0.3
            except ImportError:
                pass

        scored = [(c, scores.get(id(c), 0.0)) for c in chunks]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        return self.search(query, top_k=top_k, mode="dense")

    def ask(self, question: str, top_k: int = 5, ai_processor=None) -> str:
        results = self.retrieve(question, top_k=top_k)
        if not results:
            return "No content available to answer the question."
        context = "\n\n".join(chunk.text for chunk, _ in results)
        ai = self._get_ai(ai_processor)
        return ai.answer_question(question, context)

    async def aask(self, question: str, top_k: int = 5, ai_processor=None) -> str:
        from functools import partial
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self.ask, question, top_k=top_k, ai_processor=ai_processor))

    def ask_stream(self, question: str, top_k: int = 5, ai_processor=None):
        results = self.retrieve(question, top_k=top_k)
        if not results:
            yield "No content available to answer the question."
            return
        context = "\n\n".join(chunk.text for chunk, _ in results)
        ai = self._get_ai(ai_processor)
        yield from ai._call_stream(
            "You are a helpful assistant. Answer based solely on the provided context. "
            "If the context doesn't contain the answer, say so.",
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer in 500 words or fewer.",
            max_tokens=2000,
        )

    def compress(self, query: str, top_k: int = 5, max_tokens: int = 2000,
                 ai_processor=None) -> str:
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return ""

        texts = [chunk.text for chunk, _ in results]
        ai = self._get_ai(ai_processor)
        reranked = ai.rerank(query, texts, top_k=min(top_k, 3))
        reranked_texts = [t for t, _ in reranked]

        combined = "\n\n".join(reranked_texts)
        if ai._estimate_token_count(combined) <= max_tokens:
            return combined

        system = ("You are a context compression assistant. Extract only the sentences "
                  "that are relevant to answering the given query. Remove redundant or "
                  "irrelevant content. Preserve factual accuracy and specificity.")
        user = (f"Query: {query}\n\n"
                f"Compress the following into {max_tokens} tokens or fewer, "
                f"keeping only query-relevant information:\n\n{combined}")
        return ai._call(system, user, max_tokens=max_tokens)

    def to_llamaindex_documents(self) -> List[Any]:
        try:
            from llama_index.core import Document as LlamaindexDocument
        except ImportError:
            raise ImportError(
                "llama-index-core is required. Install with: pip install llama-index-core"
            )
        if not self._chunks:
            self.chunks()
        docs = []
        for chunk in self._chunks:
            extra = {
                "document_id": self.document_id,
                "chunk_id": chunk.chunk_id,
                "source": self.source_path or "",
                "source_type": self.source_type,
                **chunk.metadata,
            }
            docs.append(LlamaindexDocument(text=chunk.text, extra_info=extra))
        return docs

    def to_langchain_documents(self) -> List[Any]:
        try:
            from langchain_core.documents import Document as LangchainDocument
        except ImportError:
            raise ImportError(
                "langchain-core is required. Install with: pip install langchain-core"
            )
        if not self._chunks:
            self.chunks()
        docs = []
        for chunk in self._chunks:
            meta = {
                "document_id": self.document_id,
                "chunk_id": chunk.chunk_id,
                "source": self.source_path or "",
                "source_type": self.source_type,
                **chunk.metadata,
            }
            docs.append(LangchainDocument(page_content=chunk.text, metadata=meta))
        return docs

    def summary(self, max_words: int = 200, ai_processor=None) -> str:
        ai = self._get_ai(ai_processor)
        return ai.summarize(self.text, max_words=max_words)

    async def asummary(self, max_words: int = 200, ai_processor=None) -> str:
        from functools import partial
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self.summary, max_words=max_words, ai_processor=ai_processor))

    def keywords(self, top_n: int = 10, top_k: Optional[int] = None, ai_processor=None) -> List[str]:
        if top_k is not None:
            top_n = top_k
        ai = self._get_ai(ai_processor)
        return ai.extract_keywords(self.text, top_n=top_n)

    async def akeywords(self, top_n: int = 10, top_k: Optional[int] = None, ai_processor=None) -> List[str]:
        from functools import partial
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self.keywords, top_n=top_n, top_k=top_k, ai_processor=ai_processor))

    def entities(self, ai_processor=None) -> List[Dict[str, str]]:
        ai = self._get_ai(ai_processor)
        return ai.extract_entities(self.text)

    async def aentities(self, ai_processor=None) -> List[Dict[str, str]]:
        from functools import partial
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self.entities, ai_processor=ai_processor))

    def questions(self, n: int = 5, num: Optional[int] = None, ai_processor=None) -> List[str]:
        if num is not None:
            n = num
        ai = self._get_ai(ai_processor)
        return ai.generate_questions(self.text, n=n)

    async def aquestions(self, n: int = 5, num: Optional[int] = None, ai_processor=None) -> List[str]:
        from functools import partial
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self.questions, n=n, num=num, ai_processor=ai_processor))

    def flashcards(self, n: int = 10, num: Optional[int] = None, ai_processor=None) -> List[Dict[str, str]]:
        if num is not None:
            n = num
        ai = self._get_ai(ai_processor)
        return ai.generate_flashcards(self.text, n=n)

    async def aflashcards(self, n: int = 10, num: Optional[int] = None, ai_processor=None) -> List[Dict[str, str]]:
        from functools import partial
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self.flashcards, n=n, num=num, ai_processor=ai_processor))
