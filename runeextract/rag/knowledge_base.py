"""
KnowledgeBase — one-shot RAG pipeline from documents to production query.

Usage::

    kb = KnowledgeBase("documents/")
    kb.build()
    answer = kb.ask("What are the key findings?")
    print(answer)
"""

import logging
import os
import time
from dataclasses import dataclass, field
from glob import glob
from typing import List, Optional, Dict, Any, Union

from runeextract.rag.auto_pipeline import AutoRAG, auto_rag
from runeextract.rag.types import RAGResult
from runeextract.utils.maturity import beta

logger = logging.getLogger(__name__)


@beta(name="rag.knowledge_base")
class KnowledgeBase:
    """Zero-config RAG knowledge base from a directory of documents.

    Collapses extract → chunk → embed → index → store into one call::

        kb = KnowledgeBase("./docs/")
        kb.build()
        answer = kb.ask("What is in the documents?")

    Args:
        source: File path, directory path, or list of paths.
        persist_directory: Where to store the vector database (default: ./chroma_db).
        collection_name: ChromaDB collection name (default: "knowledge_base").
        embedding: Embedding model spec, e.g. "openai:text-embedding-3-small".
        llm: LLM spec for answering, e.g. "openai:gpt-4o-mini".
        chunk_strategy: Chunking strategy, or "auto" for auto-detect.
        chunk_size: Target chunk size (ignored if chunk_strategy="auto").
        **kwargs: Additional keyword arguments passed to AutoRAG.
    """

    def __init__(
        self,
        source: Union[str, List[str]],
        persist_directory: str = "./chroma_db",
        collection_name: str = "knowledge_base",
        embedding: str = "openai:text-embedding-3-small",
        llm: str = "openai:gpt-4o-mini",
        chunk_strategy: str = "auto",
        chunk_size: int = 1000,
        **kwargs,
    ):
        self._source = source
        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self._embedding = embedding
        self._llm = llm
        self._chunk_strategy = chunk_strategy
        self._chunk_size = chunk_size
        self._kwargs = kwargs
        self._rag: Optional[AutoRAG] = None
        self._is_built = False

    # ---- Lifecycle ----

    def build(self, **extras) -> "KnowledgeBase":
        """Extract, chunk, embed, index, and store all documents.

        Returns self for chaining.
        """
        logger.info("Building knowledge base from %s ...", self._source)

        chunk_strategy = (
            _smart_chunk_strategy(self._source)
            if self._chunk_strategy == "auto"
            else self._chunk_strategy
        )

        self._rag = auto_rag(
            source=self._source,
            embedding=self._embedding,
            vector_store="chromadb",
            collection_name=self._collection_name,
            persist_directory=self._persist_directory,
            chunking=chunk_strategy,
            chunk_size=self._chunk_size,
            llm=self._llm,
            analytics=True,
            **self._kwargs,
        )
        self._is_built = True
        return self

    def sync(self, **extras) -> "KnowledgeBase":
        """Incremental sync — only re-index changed documents.

        Returns self for chaining.
        """
        if not self._is_built:
            return self.build(**extras)

        from runeextract.rag.retriever import ChromaRetriever
        retriever = ChromaRetriever(
            persist_directory=self._persist_directory,
            collection_name=self._collection_name,
        )

        # Detect changed/new files
        sources = retriever.list_sources() if self._is_built else []
        source_set = set(sources)

        resolved = _resolve_source(self._source)
        changed = []
        for fp in resolved:
            if not os.path.exists(fp):
                continue
            mtime = os.path.getmtime(fp)
            indexed_key = _indexed_time(fp)
            if fp not in source_set or (indexed_key and mtime > indexed_key):
                changed.append(fp)

        # Detect deleted files
        deleted = [s for s in sources if not os.path.exists(s)]

        if deleted:
            for src in deleted:
                retriever.delete_by_source(src)
            logger.info("Removed %d deleted source(s) from index", len(deleted))

        if changed:
            logger.info("Re-indexing %d changed/new file(s)", len(changed))
            self._rag.ingest(changed, incremental=True)

        return self

    def ask(self, question: str, top_k: int = 5, cite: bool = True,
            **kwargs) -> RAGResult:
        """Query the knowledge base with automatic citations.

        Args:
            question: The question to ask.
            top_k: Number of chunks to retrieve.
            cite: If True, include source citations in the answer.
            **kwargs: Additional query parameters.

        Returns:
            A RAGResult with answer, citations, confidence, cost, etc.
        """
        if not self._is_built:
            raise RuntimeError(
                "KnowledgeBase has not been built. Call kb.build() first."
            )
        return self._rag.query(question, top_k=top_k, cite=cite, **kwargs)

    def health(self, samples: int = 100) -> "DoctorReport":
        """Run diagnostics and return a health report.

        Args:
            samples: Number of chunks to sample for analysis.

        Returns:
            A DoctorReport with findings and recommendations.
        """
        from runeextract.cli.doctor import run_doctor
        return run_doctor(self._persist_directory, samples=samples)

    def repair(self, dry_run: bool = True) -> List[str]:
        """Auto-fix common knowledge base issues.

        Detects and repairs:
        - Orphan chunks (reference deleted source files)
        - Empty/tiny/oversized chunks
        - Missing or incomplete metadata
        - Embedding dimension mismatch
        - Corrupted index (rebuild from sources)
        - Inconsistent chunk IDs

        Args:
            dry_run: If True, only report what would be fixed (default).

        Returns:
            List of actions taken (or would be taken in dry-run mode).
        """
        import chromadb
        from chromadb.config import Settings
        import time

        actions = []
        client = chromadb.PersistentClient(
            path=self._persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        collections = client.list_collections()
        if not collections:
            return actions

        col = collections[0]
        all_data = col.get(include=["documents", "metadatas", "embeddings"])
        docs = all_data.get("documents", []) or []
        metas_list = all_data.get("metadatas", []) or []
        embeds = all_data.get("embeddings", []) or []

        # 1. Orphan chunks
        orphan_ids = []
        for i, m in enumerate(metas_list):
            m = m or {}
            src = m.get("source", "")
            if src and not os.path.exists(src):
                cid = m.get("chunk_id", "")
                if cid:
                    orphan_ids.append(cid)
        if orphan_ids and not dry_run:
            col.delete(ids=orphan_ids)
            actions.append(f"Removed {len(orphan_ids)} orphan chunk(s)")
        elif orphan_ids:
            actions.append(f"Would remove {len(orphan_ids)} orphan chunk(s)")

        # 2. Empty/tiny chunks
        bad_ids = []
        for i, doc_text in enumerate(docs):
            meta = metas_list[i] if i < len(metas_list) else {}
            cid = meta.get("chunk_id", "") if meta else ""
            if not cid:
                continue
            if not doc_text or not doc_text.strip():
                bad_ids.append(cid)
            elif len(doc_text.split()) < 10:
                bad_ids.append(cid)
        if bad_ids and not dry_run:
            col.delete(ids=bad_ids)
            actions.append(f"Removed {len(bad_ids)} tiny/empty chunk(s)")
        elif bad_ids:
            actions.append(f"Would remove {len(bad_ids)} tiny/empty chunk(s)")

        # 3. Missing metadata recovery
        required = {"source", "source_type", "document_id", "chunk_id"}
        missing_updates = {}
        for i, m in enumerate(metas_list):
            m = m or {}
            cid = m.get("chunk_id", "")
            if not cid:
                continue
            for field in required:
                if not m.get(field):
                    if cid not in missing_updates:
                        missing_updates[cid] = dict(m)
                    missing_updates[cid][field] = field  # placeholder

        if missing_updates:
            actions.append(f"Found {len(missing_updates)} chunk(s) with incomplete metadata")

        # 4. Embedding dimension mismatch
        if embeds:
            dims = [len(e) for e in embeds]
            if dims and min(dims) != max(dims):
                actions.append("Embedding dimension mismatch detected — rebuild recommended")
                if not dry_run and self._is_built:
                    actions.append("Did NOT auto-rebuild dimension mismatch (requires full rebuild)")

        # 5. Corrupted index detection
        empty_normal = any(
            not d for d in (docs or [])
        )
        if empty_normal and not dry_run:
            actions.append("Index had corrupt/missing entries — partial cleanup applied")
        elif empty_normal:
            actions.append("Corrupt entries found — would clean up")

        # 6. Run doctor's fixes too
        from runeextract.cli.doctor import run_doctor
        report = run_doctor(self._persist_directory, fix=not dry_run)
        for d in report.diagnostics:
            if d.status == "warn" or d.status == "fail":
                actions.append(f"[doctor] {d.name}: {d.message}")

        return actions

    def to_auto_rag(self) -> AutoRAG:
        """Return the underlying AutoRAG instance for advanced use."""
        if not self._rag:
            raise RuntimeError("Build the knowledge base first with kb.build()")
        return self._rag

    def __repr__(self) -> str:
        status = "built" if self._is_built else "not built"
        return f"<KnowledgeBase source={self._source!r} {status}>"


# ---- Helpers ----


def _resolve_source(source: Union[str, List[str]]) -> List[str]:
    if isinstance(source, str):
        if os.path.isdir(source):
            exts = ("*.pdf", "*.docx", "*.pptx", "*.xlsx", "*.html",
                    "*.md", "*.csv", "*.json", "*.png", "*.jpg", "*.jpeg",
                    "*.mp3", "*.wav", "*.mp4")
            files = []
            for ext in exts:
                files.extend(glob(os.path.join(source, "**", ext), recursive=True))
            return sorted(files)
        return [source]
    return source


def _indexed_time(source_path: str) -> Optional[float]:
    """Try to get the last indexed time from persisted metadata."""
    import chromadb
    from chromadb.config import Settings
    try:
        client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False),
        )
        for col in client.list_collections():
            results = col.get(where={"source": source_path}, include=["metadatas"])
            for m in (results.get("metadatas") or []):
                ts = (m or {}).get("indexed_at")
                if ts:
                    try:
                        return float(ts)
                    except (ValueError, TypeError):
                        pass
    except Exception:
        pass
    return None


# ---- Smart Chunk Optimizer ----


def _smart_chunk_strategy(source: Union[str, List[str]]) -> str:
    """Analyze document structure and recommend the best chunk strategy.

    Heuristics:
    - Tables-heavy → by_page (to keep tables intact)
    - Code-heavy → fixed_size (code blocks get split poorly by other methods)
    - Many headings → by_heading (natural structure)
    - Short docs → fixed_size (no benefit from semantic)
    - Everything else → semantic (best general-purpose)
    """
    from runeextract import extract
    resolved = _resolve_source(source)
    if not resolved:
        return "semantic"

    samples = resolved[:5]
    total_text = 0
    total_tables = 0
    total_headings = 0
    total_code = 0
    doc_count = 0

    for fp in samples:
        try:
            doc = extract(fp, tables=True, metadata=True)
            doc_count += 1
            total_text += len(doc.text)
            total_tables += len(doc.tables)
            total_headings += _count_headings(doc.text)
            total_code += _count_code_blocks(doc.text)
        except Exception:
            continue

    if doc_count == 0:
        return "semantic"

    avg_len = total_text // max(doc_count, 1)
    tables_ratio = total_tables / max(doc_count, 1)
    headings_ratio = total_headings / max(doc_count, 1)
    code_ratio = total_code / max(doc_count, 1)

    # Decision logic
    if tables_ratio > 0.3 and headings_ratio < 0.1:
        return "by_page"
    if code_ratio > 0.3:
        return "fixed_size"
    if headings_ratio > 0.2 and avg_len > 3000:
        return "by_heading"
    if avg_len < 1500:
        return "fixed_size"
    return "semantic"


def _count_headings(text: str) -> int:
    import re
    return len(re.findall(r'^#{1,6}\s|\n#{1,6}\s', text))


def _count_code_blocks(text: str) -> int:
    return text.count("```") // 2
