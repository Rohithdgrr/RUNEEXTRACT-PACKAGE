"""
Zero-configuration Auto-RAG pipeline.

AutoRAG ingests documents (file, directory, URL, or list), auto-detects
the optimal chunking strategy, indexes into a vector store, and provides
a unified ``query()`` method with HyDE, multi-query expansion,
cross-encoder reranking, and cited answers.
"""

import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Callable

from runeextract import extract, extract_many
from runeextract.models.document import Document, ChunkingStrategy
from runeextract.processors.ai import AIProcessor
from runeextract.rag.types import RAGResult, Citation, ChunkWithScore
from runeextract.rag.retriever import ChromaRetriever, FAISSRetriever
from runeextract.rag.compressor import ContextualCompressor
from runeextract.rag.hierarchical import HierarchicalChunker, SummaryNode, HierarchicalResult
from runeextract.exceptions import DependencyMissingError

logger = logging.getLogger(__name__)


def auto_rag(source: Union[str, List[str]],
             embedding: str = "openai:text-embedding-3-small",
             vector_store: str = "chromadb",
             collection_name: str = "documents",
             persist_directory: str = "./chroma_db",
             chunking: str = "auto",
             chunk_size: int = 1000,
             chunk_overlap: int = 100,
             reranker: Optional[str] = None,
             llm: str = "openai:gpt-4o-mini",
             ai_processor: Optional[AIProcessor] = None,
             **extract_options) -> "AutoRAG":
    """Convenience factory: create an AutoRAG pipeline and ingest the source.

    Args:
        source: File path, directory path, URL, or list of paths.
        embedding: Embedding model spec (``provider:model_name``).
        vector_store: Vector store type (``chromadb`` or ``faiss``).
        collection_name: ChromaDB collection name.
        persist_directory: Vector store persistence directory.
        chunking: Chunking strategy (``"auto"`` for auto-detection).
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between chunks.
        reranker: Reranker spec (``"cross-encoder/ms-marco-MiniLM-L-6-v2"``).
        llm: LLM spec (``provider:model_name``).
        ai_processor: Optional shared AIProcessor instance.
        **extract_options: Passed to ``extract()``.

    Returns:
        Initialized AutoRAG instance with documents indexed.
    """
    rag = AutoRAG(embedding=embedding, vector_store=vector_store,
                  collection_name=collection_name,
                  persist_directory=persist_directory,
                  chunking=chunking, chunk_size=chunk_size,
                  chunk_overlap=chunk_overlap, reranker=reranker,
                  llm=llm, ai_processor=ai_processor)
    rag.ingest(source, **extract_options)
    return rag


class AutoRAG:
    """Zero-configuration RAG pipeline.

    Usage::

        rag = AutoRAG()
        rag.ingest("report.pdf")
        result = rag.query("What are the key findings?")
        print(result.answer)
    """

    def __init__(self,
                 embedding: str = "openai:text-embedding-3-small",
                 vector_store: str = "chromadb",
                 collection_name: str = "documents",
                 persist_directory: str = "./chroma_db",
                 chunking: str = "auto",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 100,
                 reranker: Optional[str] = None,
                 llm: str = "openai:gpt-4o-mini",
                 ai_processor: Optional[AIProcessor] = None):
        self.embedding_spec = embedding
        self.vector_store_type = vector_store
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.chunking_mode = chunking
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.reranker_spec = reranker
        self.llm_spec = llm
        self._ai = ai_processor
        self._retriever = None
        self._documents: List[Document] = []
        self._compressor = ContextualCompressor()
        self._hierarchical_chunker: Optional[HierarchicalChunker] = None

    # ------------------------------------------------------------------
    # Lazy AIProcessor initialisation
    # ------------------------------------------------------------------

    @property
    def ai(self) -> AIProcessor:
        if self._ai is not None:
            return self._ai
        llm_parts = self.llm_spec.split(":", 1)
        llm_provider = llm_parts[0] if len(llm_parts) > 1 else "openai"
        llm_model = llm_parts[1] if len(llm_parts) > 1 else llm_parts[0]
        emb_parts = self.embedding_spec.split(":", 1)
        emb_provider = emb_parts[0] if len(emb_parts) > 1 else "openai"
        #  Use the LLM provider for general AI operations; override model later
        self._ai = AIProcessor(provider=llm_provider, model=llm_model)
        self._ai._embed_provider = emb_provider
        return self._ai

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, source: Union[str, List[str]],
               **extract_options) -> List[Document]:
        """Extract, chunk, and index documents from a source.

        Args:
            source: File path, directory, URL, or list of paths.
            **extract_options: Passed to ``extract()`` / ``extract_many()``.

        Returns:
            List of ingested Document objects.
        """
        sources = self._resolve_source(source)
        logger.info(f"Ingesting {len(sources)} source(s)")

        docs = []
        for src in sources:
            try:
                doc = extract(src, **extract_options)
                docs.append(doc)
            except Exception as e:
                logger.warning(f"Skipping {src}: {e}")

        self._documents.extend(docs)
        self._chunk_and_index(docs)
        return docs

    def ingest_documents(self, documents: List[Document]) -> None:
        """Index already-extracted Document objects."""
        self._documents.extend(documents)
        self._chunk_and_index(documents)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: int = 5,
              metadata_filter: Optional[Dict[str, Any]] = None,
              return_citations: bool = True,
              hyde: bool = False,
              multi_query: bool = False,
              answer_length: str = "medium",
              **llm_kwargs) -> RAGResult:
        """End-to-end RAG query.

        Args:
            question: Natural language question.
            top_k: Number of chunks to retrieve.
            metadata_filter: Optional dict of metadata field filters.
            return_citations: Include ``[N]`` markers and citation list.
            hyde: Generate a hypothetical document for retrieval.
            multi_query: Generate 3 query variants and fuse results.
            answer_length: ``"short"``, ``"medium"``, or ``"long"``.
            **llm_kwargs: Extra kwargs for the LLM call.

        Returns:
            RAGResult with answer, citations, and metadata.
        """
        start = time.time()

        # ---- query expansion ----
        queries = [question]
        if multi_query:
            try:
                queries.extend(self.ai.expand_query(question))
            except Exception as e:
                logger.debug(f"Query expansion failed: {e}")
        if hyde:
            try:
                queries.append(self.ai.hyde(question))
            except Exception as e:
                logger.debug(f"HyDE failed: {e}")

        # ---- retrieve from all query variants ----
        all_chunks: List[ChunkWithScore] = []
        for q in queries:
            chunks = self._retrieve(q, top_k=top_k * 2, metadata_filter=metadata_filter)
            all_chunks.extend(chunks)

        unique = self._deduplicate(all_chunks)

        # ---- rerank ----
        if self.reranker_spec and len(unique) > 1:
            try:
                texts = [c.text for c in unique]
                reranked = self.ai.rerank(question, texts, top_k=top_k)
                seen = set()
                ranked_chunks = []
                for text, score in reranked:
                    for c in unique:
                        if c.text == text and id(c) not in seen:
                            c.score = score
                            seen.add(id(c))
                            ranked_chunks.append(c)
                            break
                unique = ranked_chunks[:top_k]
            except Exception as e:
                logger.debug(f"Reranking failed: {e}")
                unique = unique[:top_k]
        else:
            unique = unique[:top_k]

        # ---- compress if needed ----
        compressed = self._compressor.compress(unique, question)

        # ---- generate answer ----
        answer, citations = self._generate_answer(
            question, compressed, return_citations, answer_length, **llm_kwargs
        )

        latency = (time.time() - start) * 1000

        return RAGResult(
            answer=answer,
            citations=citations,
            confidence=self._compute_confidence(compressed),
            retrieved_chunks=compressed,
            query_variants=queries[1:] if multi_query else [],
            latency_ms=latency,
            tokens_used={
                "input": self.ai._total_input_tokens,
                "output": self.ai._total_output_tokens,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retrieve(self, query: str, top_k: int = 5,
                  metadata_filter: Optional[Dict[str, Any]] = None) -> List[ChunkWithScore]:
        """Embed the query and search the vector store.

        Uses hierarchical tree retrieval when a tree has been built.
        """
        # If we have a hierarchical tree, use it for multi-level retrieval
        if self._hierarchical_chunker is not None and self._hierarchical_chunker.tree is not None:
            result = self._hierarchical_chunker.retrieve(query, top_k=top_k)
            return [
                ChunkWithScore(
                    text=node.text,
                    score=score,
                    source=node.metadata.get("source", ""),
                    source_type="hierarchical",
                    metadata=node.metadata,
                )
                for node, score in zip(result.nodes, result.scores)
            ]

        query_embedding = self.ai.embed(query)
        if not query_embedding:
            return []
        retriever = self._get_retriever()
        if isinstance(retriever, ChromaRetriever):
            return retriever.query(query_embedding[0], top_k=top_k,
                                   metadata_filter=metadata_filter)
        return retriever.query(query_embedding[0], top_k=top_k)

    def _get_retriever(self) -> Union[ChromaRetriever, FAISSRetriever]:
        if self._retriever is not None:
            return self._retriever
        if self.vector_store_type == "chromadb":
            self._retriever = ChromaRetriever(
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
            )
        elif self.vector_store_type == "faiss":
            from runeextract.rag.retriever import FAISSRetriever
            self._retriever = FAISSRetriever(
                index_path=os.path.join(self.persist_directory, "faiss_index"),
            )
        else:
            raise ValueError(f"Unknown vector_store '{self.vector_store_type}'. "
                             f"Options: chromadb, faiss")
        return self._retriever

    def _resolve_source(self, source: Union[str, List[str]]) -> List[str]:
        if isinstance(source, str):
            p = Path(source)
            if p.is_dir():
                return [str(f) for f in p.rglob("*") if f.is_file()]
            return [source]
        return list(source)

    def _chunk_and_index(self, docs: List[Document]) -> None:
        """Chunk each document and index into the vector store."""
        for doc in docs:
            strategy = self._resolve_chunking(doc)

            if strategy == "hierarchical":
                # Create leaf chunks then build the tree
                chunks = doc.chunks(strategy=ChunkingStrategy.HIERARCHICAL,
                                    size=self.chunk_size,
                                    overlap=self.chunk_overlap,
                                    leaf_size=300)
                if chunks:
                    self._build_hierarchical_tree(doc, chunks)
                continue

            chunks = doc.chunks(strategy=strategy, size=self.chunk_size,
                                overlap=self.chunk_overlap)
            if not chunks:
                continue
            # Index based on store type
            if self.vector_store_type == "chromadb":
                try:
                    doc.to_chromadb(
                        collection_name=self.collection_name,
                        persist_directory=self.persist_directory,
                    )
                except ImportError:
                    logger.warning("chromadb not installed; skipping index")
            elif self.vector_store_type == "faiss":
                try:
                    doc.to_faiss(
                        index_path=os.path.join(self.persist_directory, "faiss_index"),
                    )
                except ImportError:
                    logger.warning("faiss not installed; skipping index")

    def _build_hierarchical_tree(self, doc: Document, chunks: list):
        """Build RAPTOR-style hierarchical tree from leaf chunks."""
        from runeextract.rag.hierarchical import HierarchicalChunker

        texts = [c.text for c in chunks]
        metadata_list = [c.metadata for c in chunks]

        # Build summarizer using AIProcessor if available
        summarizer = None
        if hasattr(self, 'ai') and self.ai is not None:
            def make_summarizer(ai):
                def _summarize(texts: List[str]) -> str:
                    combined = "\n\n".join(texts)
                    if len(combined) > 8000:
                        combined = combined[:8000]
                    prompt = (
                        "Summarize the following text concisely, preserving "
                        "key facts, names, numbers, and relationships."
                    )
                    try:
                        return ai._call(prompt, combined)
                    except Exception:
                        return combined[:1000]
                return _summarize
            summarizer = make_summarizer(self.ai)

        chunker = HierarchicalChunker(
            summarizer=summarizer,
            cluster_size=5,
            max_levels=3,
        )
        tree = chunker.build_tree(texts, metadata_list)

        # Index all nodes at all levels into vector store
        all_nodes = [tree]
        queue = list(tree.children)
        while queue:
            node = queue.pop(0)
            all_nodes.append(node)
            queue.extend(node.children)

        for node in all_nodes:
            dummy_doc = Document(
                text=node.text,
                metadata={
                    "level": node.level,
                    "node_id": node.node_id,
                    "num_children": len(node.children),
                    "strategy": "hierarchical",
                    "source": doc.source_path or "",
                },
            )
            dummy_doc._chunks = [
                type('Chunk', (), {
                    'text': node.text,
                    'metadata': {
                        'level': node.level,
                        'node_id': node.node_id,
                        'num_children': len(node.children),
                        'strategy': 'hierarchical',
                        'source': doc.source_path or '',
                    }
                })()
            ]
            try:
                if self.vector_store_type == "chromadb":
                    dummy_doc.to_chromadb(
                        collection_name=self.collection_name,
                        persist_directory=self.persist_directory,
                    )
                elif self.vector_store_type == "faiss":
                    dummy_doc.to_faiss(
                        index_path=os.path.join(self.persist_directory, "faiss_index"),
                    )
            except ImportError:
                pass

        self._hierarchical_chunker = chunker
        logger.info(f"Built hierarchical tree: {len(all_nodes)} nodes across {max(n.level for n in all_nodes) + 1} levels")

    def _resolve_chunking(self, doc: Document) -> str:
        """Auto-detect optimal chunking strategy or return configured one."""
        if self.chunking_mode != "auto":
            return self.chunking_mode

        text = doc.text[:3000]
        ext = (doc.source_path or "").lower()

        code_exts = {".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs", ".rb"}
        if any(ext.endswith(e) for e in code_exts):
            return "by_heading"

        if "##" in text or "# " in text:
            return "by_heading"

        if len(doc.tables) > 3:
            return "fixed_size"

        words = len(doc.text.split())
        if words > 10000:
            return "hierarchical"

        academic = {"introduction", "methods", "results", "conclusion",
                    "abstract", "references"}
        if academic & set(text.lower().split()[:50]):
            return "by_heading"

        return "sentence_window"

    def _deduplicate(self, chunks: List[ChunkWithScore]) -> List[ChunkWithScore]:
        seen = set()
        unique = []
        for c in chunks:
            key = (c.text[:100], c.source)
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique

    def _generate_answer(self, question: str, chunks: List[ChunkWithScore],
                         return_citations: bool, length: str,
                         **llm_kwargs) -> tuple:
        """Build context with citation markers and call the LLM."""
        import re

        context_parts = []
        for i, c in enumerate(chunks, 1):
            src = c.source or "unknown"
            page = f" p.{c.page}" if c.page is not None else ""
            context_parts.append(
                f"[{i}] Source: {src}{page}\n{c.text}"
            )
        context = "\n---\n".join(context_parts)

        length_map = {"short": "2-3 sentences",
                      "medium": "1 paragraph",
                      "long": "3-5 paragraphs"}
        length_inst = length_map.get(length, "1 paragraph")

        prompt = (
            f"Answer the question using ONLY the provided context. "
            f"Cite sources using [1], [2], etc. "
            f"If the context doesn't contain the answer, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Requirements:\n"
            f"- Length: {length_inst}\n"
            f"- Cite every factual claim with [number]\n"
            f"- Be concise and accurate"
        )

        answer = self.ai._call(
            "You are a helpful RAG assistant. Answer accurately with citations.",
            prompt,
            **llm_kwargs,
        )

        citations = []
        if return_citations:
            cited_nums = set()
            for m in re.finditer(r'\[(\d+)\]', answer):
                cited_nums.add(int(m.group(1)))
            for n in cited_nums:
                idx = n - 1
                if 0 <= idx < len(chunks):
                    c = chunks[idx]
                    citations.append(Citation(
                        text=c.text[:300],
                        source=c.source,
                        page=c.page,
                        chunk_index=idx,
                        relevance_score=c.score,
                    ))

        return answer, citations

    def _compute_confidence(self, chunks: List[ChunkWithScore]) -> float:
        if not chunks:
            return 0.0
        return float(sum(c.score for c in chunks)) / len(chunks)
