"""
Document model for unified extraction output.
"""

import json
import re
import uuid
import logging
import struct
import random
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

_TOKEN_ENCODING_CACHE = {}


def _get_token_encoding(encoding_name: str = "cl100k_base"):
    """Get a tiktoken encoding, caching it for reuse. Returns None if tiktoken not installed."""
    if encoding_name not in _TOKEN_ENCODING_CACHE:
        try:
            import tiktoken
            _TOKEN_ENCODING_CACHE[encoding_name] = tiktoken.get_encoding(encoding_name)
        except ImportError:
            _TOKEN_ENCODING_CACHE[encoding_name] = None
    return _TOKEN_ENCODING_CACHE.get(encoding_name)


def _estimate_token_count(text: str) -> int:
    """Rough token estimate (~4 chars per token) when tiktoken is not available."""
    return max(1, len(text) // 4)


class ChunkingStrategy(str, Enum):
    """Chunking strategies for document processing."""
    BY_PAGE = "by_page"
    BY_HEADING = "by_heading"
    SEMANTIC = "semantic"
    FIXED_SIZE = "fixed_size"
    BY_TOKEN = "by_token"


@dataclass
class Image:
    """Represents an extracted image."""
    data: bytes
    format: str
    width: Optional[int] = None
    height: Optional[int] = None
    page_number: Optional[int] = None
    caption: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Table:
    """Represents an extracted table."""
    rows: List[List[str]]
    columns: List[str]
    page_number: Optional[int] = None
    caption: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self):
        """Convert table to pandas DataFrame."""
        try:
            import pandas as pd
            return pd.DataFrame(self.rows, columns=self.columns)
        except ImportError:
            raise ImportError("pandas is required for DataFrame conversion. Install with: pip install pandas")


@dataclass
class Chunk:
    """Represents a chunk of text for RAG applications."""
    text: str
    chunk_id: str
    start_index: int
    end_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_document_id: Optional[str] = None

    def token_count(self, encoding_name: str = "cl100k_base") -> int:
        """Number of tokens in the chunk text using the specified encoding."""
        enc = _get_token_encoding(encoding_name)
        if enc:
            return len(enc.encode(self.text))
        return _estimate_token_count(self.text)


@dataclass
class Document:
    """
    Universal document model for all extraction types.
    
    No matter the source (PDF, DOCX, HTML, etc.), the output schema is identical.
    """
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
        """Number of tokens in the full document text using the specified encoding."""
        enc = _get_token_encoding(encoding_name)
        if enc:
            return len(enc.encode(self.text))
        return _estimate_token_count(self.text)

    def chunks(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE,
        size: int = 1000,
        overlap: int = 100,
        on_chunk: Optional[Callable[['Chunk'], None]] = None,
        **kwargs
    ) -> List[Chunk]:
        """
        Chunk the document text for RAG applications.
        
        Args:
            strategy: Chunking strategy (by_page, by_heading, semantic, fixed_size, by_token)
            size: Target chunk size (characters for most strategies, tokens for by_token)
            overlap: Overlap between chunks (characters or tokens)
            on_chunk: Optional callback invoked for each chunk as it is created
            **kwargs: Additional strategy-specific parameters
                - encoding_name: tiktoken encoding name (default "cl100k_base", for by_token)
            
        Returns:
            List of Chunk objects
        """
        params = {"strategy": strategy, "size": size, "overlap": overlap, **kwargs}
        if self._chunks is not None and self._chunk_params == params:
            return self._chunks

        self._chunk_params = params

        if strategy == ChunkingStrategy.FIXED_SIZE:
            self._chunks = self._chunk_fixed_size(size, overlap)
        elif strategy == ChunkingStrategy.BY_PAGE:
            self._chunks = self._chunk_by_page(**kwargs)
        elif strategy == ChunkingStrategy.BY_HEADING:
            self._chunks = self._chunk_by_heading(**kwargs)
        elif strategy == ChunkingStrategy.SEMANTIC:
            self._chunks = self._chunk_semantic(size, **kwargs)
        elif strategy == ChunkingStrategy.BY_TOKEN:
            self._chunks = self._chunk_by_token(size, overlap, **kwargs)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        for chunk in self._chunks:
            chunk.parent_document_id = self.document_id

        if on_chunk:
            for chunk in self._chunks:
                on_chunk(chunk)

        return self._chunks

    def _chunk_fixed_size(self, size: int, overlap: int) -> List[Chunk]:
        """Chunk text by fixed size with overlap."""
        chunks = []
        text = self.text
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + size
            chunk_text = text[start:end]
            
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=f"chunk_{chunk_id}",
                start_index=start,
                end_index=end,
                metadata={"strategy": "fixed_size", "size": size, "overlap": overlap}
            ))
            
            step = max(size - overlap, 1)
            start += step
            chunk_id += 1

        return chunks

    def _chunk_by_page(self, **kwargs) -> List[Chunk]:
        """Chunk text by page boundaries using page_breaks from metadata."""
        page_breaks = self.metadata.get("page_breaks", [])
        if not page_breaks:
            return [Chunk(
                text=self.text,
                chunk_id="chunk_0",
                start_index=0,
                end_index=len(self.text),
                metadata={"strategy": "by_page"}
            )]

        chunks = []
        prev = 0
        for idx, break_pos in enumerate(page_breaks):
            chunk_text = self.text[prev:break_pos]
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=f"chunk_{idx}",
                start_index=prev,
                end_index=break_pos,
                metadata={"strategy": "by_page", "page": idx + 1}
            ))
            prev = break_pos
        if prev < len(self.text):
            chunks.append(Chunk(
                text=self.text[prev:],
                chunk_id=f"chunk_{len(chunks)}",
                start_index=prev,
                end_index=len(self.text),
                metadata={"strategy": "by_page", "page": len(chunks) + 1}
            ))
        return chunks

    def _chunk_by_heading(self, **kwargs) -> List[Chunk]:
        """Chunk text by heading structure (Markdown-style # headings, === underlines)."""
        # Patterns: # heading, ## heading, === underline, --- underline
        heading_pattern = re.compile(
            r'^(#{1,6}\s+.*)$|^(.+)\n[=\-]+\s*$',
            re.MULTILINE
        )
        matches = list(heading_pattern.finditer(self.text))
        if not matches:
            return self._chunk_fixed_size(1000, 0)

        chunks = []
        prev_end = 0
        for idx, match in enumerate(matches):
            heading_text = (match.group(1) or match.group(2)).strip()
            if match.start() > prev_end:
                chunk_text = self.text[prev_end:match.start()]
                if chunk_text.strip():
                    chunks.append(Chunk(
                        text=chunk_text.strip(),
                        chunk_id=f"chunk_{idx}",
                        start_index=prev_end,
                        end_index=match.start(),
                        metadata={"strategy": "by_heading"}
                    ))
            prev_end = match.start()
        if prev_end < len(self.text):
            chunks.append(Chunk(
                text=self.text[prev_end:].strip(),
                chunk_id=f"chunk_{len(chunks)}",
                start_index=prev_end,
                end_index=len(self.text),
                metadata={"strategy": "by_heading"}
            ))
        return chunks or [Chunk(
            text=self.text,
            chunk_id="chunk_0",
            start_index=0,
            end_index=len(self.text),
            metadata={"strategy": "by_heading"}
        )]

    def _chunk_semantic(self, size: int, **kwargs) -> List[Chunk]:
        """Chunk text using semantic boundaries (sentences, paragraphs)."""
        if self.text.strip():
            sep = "_PARAGRAPH_BREAK_"
            text_with_markers = self.text.replace('\n\n', f'\n{sep}\n')
        else:
            text_with_markers = self.text
        parts = text_with_markers.split(sep)
        chunks = []
        current_chunk = ""
        chunk_id = 0
        start_index = 0
        for part in parts:
            part = part.strip()
            if not part:
                continue
            part_len = len(part)
            if len(current_chunk) + part_len > size and current_chunk:
                chunks.append(Chunk(
                    text=current_chunk.strip(),
                    chunk_id=f"chunk_{chunk_id}",
                    start_index=start_index,
                    end_index=start_index + len(current_chunk),
                    metadata={"strategy": "semantic"}
                ))
                start_index += len(current_chunk)
                current_chunk = part
                chunk_id += 1
            else:
                if current_chunk:
                    current_chunk += "\n\n" + part
                else:
                    current_chunk = part
        if current_chunk:
            chunks.append(Chunk(
                text=current_chunk.strip(),
                chunk_id=f"chunk_{chunk_id}",
                start_index=start_index,
                end_index=start_index + len(current_chunk),
                metadata={"strategy": "semantic"}
            ))
        return chunks if chunks else [Chunk(
            text=self.text,
            chunk_id="chunk_0",
            start_index=0,
            end_index=len(self.text),
            metadata={"strategy": "semantic"}
        )]

    def _chunk_by_token(self, size: int, overlap: int, encoding_name: str = "cl100k_base", **kwargs) -> List[Chunk]:
        """Chunk text by token count using tiktoken (falls back to char estimate)."""
        enc = _get_token_encoding(encoding_name)
        text = self.text
        chunks = []
        chunk_id = 0
        start = 0

        if enc:
            tokens = enc.encode(text)
            token_count = len(tokens)
            while start < token_count:
                end = min(start + size, token_count)
                chunk_tokens = tokens[start:end]
                chunk_text = enc.decode(chunk_tokens)
                char_start = len(enc.decode(tokens[:start])) if start > 0 else 0
                char_end = char_start + len(chunk_text)
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=f"chunk_{chunk_id}",
                    start_index=char_start,
                    end_index=char_end,
                    metadata={
                        "strategy": "by_token",
                        "size": size,
                        "overlap": overlap,
                        "encoding": encoding_name,
                        "token_start": start,
                        "token_end": end,
                    }
                ))
                step = max(size - overlap, 1)
                start += step
                chunk_id += 1
        else:
            chunks = self._chunk_fixed_size(size * 4, overlap * 4)

        return chunks

    def _clean_str(self, val: Any) -> Any:
        """Recursively strip null bytes from strings in a data structure."""
        if isinstance(val, str):
            return val.replace("\x00", "")
        if isinstance(val, dict):
            return {k: self._clean_str(v) for k, v in val.items()}
        if isinstance(val, list):
            return [self._clean_str(v) for v in val]
        return val

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary representation."""
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
                    "metadata": chunk.metadata
                }
                for chunk in (self._chunks or [])
            ]
        })

    def to_json(self, indent: int = 2) -> str:
        """Serialize document to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        """Convert extracted content to Markdown format."""
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
        """Number of words in the document text."""
        return len(self.text.split())

    @property
    def text_length(self) -> int:
        """Number of characters in the document text."""
        return len(self.text)

    @classmethod
    def merge(cls, documents: List["Document"]) -> "Document":
        """Merge multiple documents into one. Text is concatenated with newlines."""
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

    def to_openai_messages(self, system_message: Optional[str] = None, include_all: bool = True) -> List[Dict[str, str]]:
        """Convert document chunks to OpenAI chat message format.
        
        Args:
            system_message: Optional system message to prepend
            include_all: If True, include all chunks. If False, include only the full text.
            
        Returns:
            List of message dicts compatible with OpenAI chat completion API
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        if include_all and self._chunks:
            for chunk in self._chunks:
                messages.append({"role": "user", "content": chunk.text})
        else:
            messages.append({"role": "user", "content": self.text})
        return messages

    def to_chromadb(self, collection_name: str = "documents", persist_directory: str = "./chroma_db", embedding_function=None,
                    on_insert: Optional[Callable[[str], None]] = None):
        """Store document chunks in a ChromaDB collection (optional dep).
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_function: ChromaDB embedding function (uses default if None)
            on_insert: Optional callback invoked with chunk_id after each insert
            
        Returns:
            ChromaDB Collection object
        """
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
            **chunk.metadata,
        } for chunk in self._chunks]
        collection.add(documents=texts, ids=ids, metadatas=metadatas)
        if on_insert:
            for cid in ids:
                on_insert(cid)
        return collection

    def to_faiss(self, index_path: str = "./faiss_index", embedding_dim: int = 384):
        """Store document chunks in a FAISS index (optional dep).
        
        Args:
            index_path: Path to save the FAISS index and metadata
            embedding_dim: Dimension of embeddings (default 384 for sentence-transformers/all-MiniLM-L6-v2)
            
        Returns:
            Tuple of (FAISS index, list of metadata dicts)
        """
        try:
            import faiss
            import numpy as np
        except ImportError:
            raise ImportError(
                "faiss and numpy are required. Install with: pip install faiss-cpu numpy"
            )
        import pickle
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
            **chunk.metadata,
        } for chunk in self._chunks]

        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
        faiss.write_index(index, index_path + ".index")
        with open(index_path + ".meta.pkl", "wb") as f:
            pickle.dump(metadata_list, f)

        return index, metadata_list

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """Retrieve chunks most relevant to a query using embedding similarity.

        Args:
            query: Natural language query
            top_k: Number of chunks to return

        Returns:
            List of (Chunk, score) tuples sorted by relevance (highest first)
        """
        if not self._chunks:
            self.chunks()
        if not self._chunks:
            return []

        from runeextract.processors.ai import AIProcessor
        ai = AIProcessor()
        query_embedding = ai.embed(query)[0]
        import numpy as np
        chunk_texts = [c.text for c in self._chunks]
        chunk_embeddings = ai.embed(chunk_texts)

        query_vec = np.array(query_embedding, dtype=np.float32)
        chunk_vecs = np.array(chunk_embeddings, dtype=np.float32)
        norms = np.linalg.norm(chunk_vecs, axis=1) * np.linalg.norm(query_vec)
        scores = np.dot(chunk_vecs, query_vec) / np.maximum(norms, 1e-10)

        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(self._chunks[i], float(scores[i])) for i in top_indices]

    def ask(self, question: str, top_k: int = 5) -> str:
        """Ask a question about the document using RAG (retrieve + answer).

        Args:
            question: Natural language question
            top_k: Number of chunks to retrieve as context

        Returns:
            Answer string based on document content
        """
        results = self.retrieve(question, top_k=top_k)
        if not results:
            return "No content available to answer the question."
        context = "\n\n".join(chunk.text for chunk, _ in results)
        from runeextract.processors.ai import AIProcessor
        ai = AIProcessor()
        return ai.answer_question(question, context)

    def to_llamaindex_documents(self) -> List[Any]:
        """Convert document chunks to LlamaIndex Document objects (optional dep).

        Returns:
            List of llama_index.core.Document objects
        """
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

    def summary(self, max_words: int = 200) -> str:
        """AI-powered text summary (requires openai + OPENAI_API_KEY)."""
        from runeextract.processors.ai import AIProcessor
        ai = AIProcessor()
        return ai.summarize(self.text, max_words=max_words)

    def keywords(self, top_n: int = 10) -> List[str]:
        """AI-powered keyword extraction."""
        from runeextract.processors.ai import AIProcessor
        ai = AIProcessor()
        return ai.extract_keywords(self.text, top_n=top_n)

    def entities(self) -> List[Dict[str, str]]:
        """AI-powered named entity extraction."""
        from runeextract.processors.ai import AIProcessor
        ai = AIProcessor()
        return ai.extract_entities(self.text)

    def questions(self, n: int = 5) -> List[str]:
        """AI-powered question generation."""
        from runeextract.processors.ai import AIProcessor
        ai = AIProcessor()
        return ai.generate_questions(self.text, n=n)

    def flashcards(self, n: int = 10) -> List[Dict[str, str]]:
        """AI-powered flashcard generation."""
        from runeextract.processors.ai import AIProcessor
        ai = AIProcessor()
        return ai.generate_flashcards(self.text, n=n)
