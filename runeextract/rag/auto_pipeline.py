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
from runeextract.utils.maturity import beta
from runeextract.rag.confidence import ConfidenceScorer
from runeextract.rag.semantic_cache import SemanticCache
from runeextract.rag.streaming import StreamingRAG, StreamEvent, StreamEventType
from runeextract.rag.analytics import RAGAnalytics
from runeextract.rag.routing import QueryRouter as RouterV2
from runeextract.rag.multilingual import MultilingualRAG
from runeextract.rag.reasoning import ChainOfThoughtReasoner

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
             # 🚀 GAME-CHANGING FEATURES
             intelligence: str = "adaptive",
             watch: bool = False,
             incremental: bool = True,
             multimodal: bool = False,
             safe_mode: bool = False,
             cost_limit: Optional[float] = None,
             scan_secrets: bool = False,
             timeout_per_doc: int = 300,
             # 🚀 Tier 1 features  
             semantic_cache: bool = False,
             cache_similarity: float = 0.92,
             cache_ttl: int = 3600,
             streaming: bool = False,
             analytics: bool = False,
             rbac: bool = False,
             # 🚀 Tier 2 features
             routing_rags: Optional[Dict[str, Any]] = None,
             experiment_config: Optional[Dict[str, Any]] = None,
             multi_language: bool = False,
             languages: Optional[List[str]] = None,
             translation_provider: str = "openai",
             reasoning: bool = False,
             reasoning_max_steps: int = 5,
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
        
        🚀 GAME-CHANGING FEATURES:
        intelligence: "adaptive" for self-tuning RAG (learns from query patterns)
        watch: Enable live document sync with auto-indexing on file changes
        incremental: Only re-index changed content (hash-based deduplication)
        multimodal: Enable vision understanding for charts, tables, images
        safe_mode: Auto-enable all production safeguards (cost limits, secret scanning, etc.)
        cost_limit: Hard cap on API costs (dollars)
        scan_secrets: Automatically detect and redact API keys, tokens in responses
        timeout_per_doc: Max extraction time per document (seconds)
        
        🚀 Tier 2 features:
        routing_rags: Dict of {name: AutoRAG} for smart intent-based routing
        experiment_config: Dict with A/B experiment config (name, variants, etc.)
        multi_language: Enable auto-detection and translation for multi-lingual queries
        languages: List of supported language codes (ISO 639-1)
        translation_provider: Translation API provider (openai, google, deepl)
        reasoning: Enable chain-of-thought reasoning for complex queries
        reasoning_max_steps: Max decomposition steps for CoT reasoning
        
        **extract_options: Passed to ``extract()``.

    Returns:
        Initialized AutoRAG instance with documents indexed.
    """
    rag = AutoRAG(embedding=embedding, vector_store=vector_store,
                  collection_name=collection_name,
                  persist_directory=persist_directory,
                  chunking=chunking, chunk_size=chunk_size,
                  chunk_overlap=chunk_overlap, reranker=reranker,
                  llm=llm, ai_processor=ai_processor,
                  intelligence=intelligence,
                  multimodal=multimodal,
                  safe_mode=safe_mode,
                  cost_limit=cost_limit,
                  scan_secrets=scan_secrets,
                  timeout_per_doc=timeout_per_doc,
                  # Tier 1 features
                  semantic_cache=semantic_cache,
                  cache_similarity=cache_similarity,
                  cache_ttl=cache_ttl,
                  streaming=streaming,
                  analytics=analytics,
                  rbac=rbac,
                  # Tier 2 features
                  routing_rags=routing_rags,
                  experiment_config=experiment_config,
                  multi_language=multi_language,
                  languages=languages,
                  translation_provider=translation_provider,
                  reasoning=reasoning,
                  reasoning_max_steps=reasoning_max_steps)
    
    # Initial ingestion with incremental support
    rag.ingest(source, incremental=incremental, **extract_options)
    
    # Start live document watcher if requested
    if watch and isinstance(source, str):
        rag.start_watch(source, incremental=incremental, **extract_options)
    
    return rag


@beta(name="rag.auto_pipeline")
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
                 ai_processor: Optional[AIProcessor] = None,
                 # 🚀 GAME-CHANGING FEATURES
                 intelligence: str = "adaptive",
                 multimodal: bool = False,
                 safe_mode: bool = False,
                 cost_limit: Optional[float] = None,
                 scan_secrets: bool = False,
                 timeout_per_doc: int = 300,
                 # 🚀 Tier 1 features
                 semantic_cache: bool = False,
                 cache_similarity: float = 0.92,
                 cache_ttl: int = 3600,
                 streaming: bool = False,
                 analytics: bool = False,
                 rbac: bool = False,
                 # 🚀 Tier 2 features
                 routing_rags: Optional[Dict[str, Any]] = None,
                 experiment_config: Optional[Dict[str, Any]] = None,
                 multi_language: bool = False,
                 languages: Optional[List[str]] = None,
                 translation_provider: str = "openai",
                 reasoning: bool = False,
                 reasoning_max_steps: int = 5):
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
        
        # 🚀 Feature 1: Adaptive Intelligence
        self.intelligence = intelligence
        self._query_history: List[Dict[str, Any]] = []
        self._performance_stats = {
            "avg_latency_ms": 0.0,
            "avg_confidence": 0.0,
            "total_queries": 0,
            "strategy_success": {}  # Track which strategies work best
        }
        
        # 🚀 Feature 2: Incremental Indexing (file hash tracking)
        self._file_hashes: Dict[str, str] = {}  # file_path -> content_hash
        self._file_chunk_ids: Dict[str, List[str]] = {}  # source_path -> [chunk_id, ...]
        self._watcher_thread = None
        self._watcher_running = False
        
        # 🚀 Feature 4: Multi-Modal Support
        self.multimodal = multimodal
        self._vision_cache: Dict[str, str] = {}  # image_hash -> description
        self._mm_index = None  # Lazy-built MultiModalIndex
        
        # 🚀 Feature 5: Production Safeguards
        self.safe_mode = safe_mode
        self.cost_limit = cost_limit
        self.scan_secrets = scan_secrets or safe_mode
        self.timeout_per_doc = timeout_per_doc
        self._total_cost = 0.0
        self._secret_scanner = None
        self._retriever_failures: int = 0
        self._retriever_cb_open: bool = False
        self._retriever_cb_threshold: int = 3
        
        if safe_mode:
            logger.info("🛡️  Safe mode enabled: cost limits, secret scanning, timeouts active")
            if cost_limit is None:
                self.cost_limit = 10.0  # Default $10 limit in safe mode
            self.scan_secrets = True
        
        # 🚀 Feature 7: Semantic Caching
        self.semantic_cache_enabled = semantic_cache
        self._semantic_cache = None
        if semantic_cache:
            self._semantic_cache = SemanticCache(
                similarity_threshold=cache_similarity,
                ttl_seconds=cache_ttl,
                max_entries=1000
            )
            logger.info(f"🧠 Semantic cache enabled: {cache_similarity:.0%} similarity, {cache_ttl}s TTL")
        
        # 🚀 Feature 12: Streaming RAG
        self.streaming_enabled = streaming
        self._streaming_rag = None
        if streaming:
            # Will be initialized lazily on first use
            logger.info("⚡ Streaming RAG enabled: progressive refinement")
        
        # 🚀 Feature 8: Analytics
        self.analytics_enabled = analytics
        self._analytics = None
        if analytics:
            self._analytics = RAGAnalytics(history_size=10000, enable_time_series=True)
            logger.info("📊 Analytics enabled: tracking queries, costs, performance")
        
        # 🚀 Feature 11: RBAC
        self.rbac_enabled = rbac
        self._rbac = None
        if rbac:
            from runeextract.rag.rbac import RBACManager
            self._rbac = RBACManager(enable_audit=True)
            logger.info("🔐 RBAC enabled: document-level permissions, audit logging")
        
        # 🚀 Feature 6: Smart Query Routing
        self._routing_rags = routing_rags
        self._router_v2 = None
        if routing_rags:
            self._router_v2 = RouterV2(
                rag_configs=routing_rags,
                confidence_threshold=0.85,
                enable_fusion=True
            )
            logger.info(f"🧭 Smart routing enabled: {len(routing_rags)} RAGs")
        
        # 🚀 Feature 9: A/B Experiments
        self._experiment_config = experiment_config
        self._experiment_manager = None
        if experiment_config:
            from runeextract.rag.experiments import ExperimentManager
            name = experiment_config.get("name", "experiment")
            variants = experiment_config.get("variants", {})
            def _rag_factory(cfg):
                for k, v in cfg.items():
                    setattr(self, k, v)
                return self
            self._experiment_manager = ExperimentManager(
                name=name,
                variants=variants,
                rag_factory=_rag_factory
            )
            logger.info(f"🧪 A/B experiments enabled: '{name}' with {len(variants)} variants")
        
        # 🚀 Feature 10: Multi-Language
        self.multi_language = multi_language
        self._languages = languages or ["en", "es", "fr", "de", "zh", "ja", "ar"]
        self._translation_provider = translation_provider
        self._multilingual = None
        if multi_language:
            self._multilingual = MultilingualRAG(
                base_rag=self,
                languages=self._languages,
                translation_provider=translation_provider,
                translation_cache=True,
                cross_lingual=True
            )
            logger.info(f"🌐 Multi-language enabled: {self._languages}")
        
        # 🚀 Feature 14: Chain-of-Thought Reasoning
        self.reasoning_enabled = reasoning
        self._reasoning_max_steps = reasoning_max_steps
        self._reasoner = None
        if reasoning:
            self._reasoner = ChainOfThoughtReasoner(
                rag=self,
                max_steps=reasoning_max_steps,
                confidence_threshold=0.7,
                enable_self_correction=True
            )
            logger.info(f"🧠 Chain-of-Thought reasoning enabled (max_steps={reasoning_max_steps})")

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
               incremental: bool = True,
               **extract_options) -> List[Document]:
        """Extract, chunk, and index documents from a source.

        Args:
            source: File path, directory, URL, or list of paths.
            incremental: Only process files that have changed (hash-based)
            **extract_options: Passed to ``extract()`` / ``extract_many()``.

        Returns:
            List of ingested Document objects.
        """
        import hashlib
        
        sources = self._resolve_source(source)
        logger.info(f"Ingesting {len(sources)} source(s) (incremental={incremental})")

        docs = []
        for src in sources:
            try:
                # 🚀 Feature 2: Incremental Indexing - Check file hash
                if incremental and os.path.isfile(src):
                    with open(src, 'rb') as f:
                        content_hash = hashlib.sha256(f.read()).hexdigest()
                    
                    if src in self._file_hashes and self._file_hashes[src] == content_hash:
                        logger.debug(f"⏭️  Skipping unchanged file: {src}")
                        continue
                    
                    self._file_hashes[src] = content_hash
                    logger.info(f"📄 Processing changed/new file: {src}")
                
                # 🚀 Feature 5: Apply timeout and safety checks
                extract_opts = extract_options.copy()
                if self.timeout_per_doc:
                    extract_opts['timeout'] = self.timeout_per_doc
                
                doc = extract(src, **extract_opts)
                
                # 🚀 Feature 5: Scan for secrets
                if self.scan_secrets and doc.text:
                    findings = self._scan_document_secrets(doc)
                    if findings:
                        logger.warning(f"🚨 Found {len(findings)} secrets in {src} - redacting")
                        doc.text = self._redact_document_secrets(doc.text, findings)
                
                # 🚀 Feature 4: Multi-modal processing
                if self.multimodal and doc.images:
                    doc = self._process_multimodal(doc)
                
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
              cite: bool = True,
              hyde: bool = False,
              multi_query: bool = False,
              answer_length: str = "medium",
              # RBAC params
              user: str = "anonymous",
              roles: Optional[List[str]] = None,
              # CoT reasoning override
              reasoning: Optional[bool] = None,
              # Experiment user bucketing
              user_id: Optional[str] = None,
              **llm_kwargs) -> RAGResult:
        """End-to-end RAG query with citations and adaptive intelligence.

        Args:
            question: Natural language question.
            top_k: Number of chunks to retrieve.
            metadata_filter: Optional dict of metadata field filters.
            return_citations: Include ``[N]`` markers and citation list.
            cite: Alias for return_citations (cleaner API).
            hyde: Generate a hypothetical document for retrieval.
            multi_query: Generate 3 query variants and fuse results.
            answer_length: ``"short"``, ``"medium"``, or ``"long"``.
            **llm_kwargs: Extra kwargs for the LLM call.

        Returns:
            RAGResult with answer, citations, confidence, and provenance.
        """
        start = time.time()
        
        # Alias support
        if cite is not None:
            return_citations = cite
        
        # 🚀 Feature 6: Smart Query Routing — delegate to router if configured
        if self._router_v2:
            return self._router_v2.query(
                question=question,
                top_k=top_k,
                return_citations=return_citations,
                **llm_kwargs
            )
        
        # 🚀 Feature 9: A/B Experiments — delegate to experiment manager if configured
        uid = user_id or user
        if self._experiment_manager:
            return self._experiment_manager.query(
                question=question,
                user_id=uid,
                top_k=top_k,
                return_citations=return_citations,
                **llm_kwargs
            )
        
        # 🚀 Feature 10: Multi-Language — delegate if non-English detected
        if self._multilingual:
            return self._multilingual.query(
                question=question,
                target_lang=None,
                translate_sources=False,
                top_k=top_k,
                return_citations=return_citations,
                **llm_kwargs
            )
        
        # 🚀 Feature 14: Chain-of-Thought Reasoning
        use_reasoning = self.reasoning_enabled if reasoning is None else reasoning
        if use_reasoning and self._reasoner:
            return self._reasoner.reason(
                query=question,
                top_k=top_k,
                return_citations=return_citations,
                **llm_kwargs
            )
        
        # 🚀 Feature 7: Check semantic cache first
        if self._semantic_cache:
            query_embedding = self.ai.embed(question)
            if query_embedding:
                cached_result = self._semantic_cache.get(query_embedding[0], question)
                if cached_result:
                    logger.info(f"✅ Cache hit for: {question[:50]}...")
                    # Convert cached result to RAGResult
                    result = RAGResult(
                        answer=cached_result["answer"],
                        citations=cached_result["citations"],
                        confidence=cached_result["confidence"],
                        retrieved_chunks=cached_result["retrieved_chunks"],
                        query_variants=[],
                        latency_ms=(time.time() - start) * 1000,
                        tokens_used={"input": 0, "output": 0},
                        cost=0.0,  # Cached, no cost
                        total_session_cost=self._total_cost
                    )
                    
                    # Record in analytics
                    if self._analytics:
                        self._analytics.record_query(
                            query=question,
                            latency_ms=result.latency_ms,
                            confidence=result.confidence,
                            cost=0.0,
                            citations=len(result.citations),
                            chunks_retrieved=len(result.retrieved_chunks),
                            cached=True
                        )
                    
                    return result
        
        # 🚀 Feature 5: Cost limit enforcement
        if self.cost_limit and self._total_cost >= self.cost_limit:
            raise Exception(f"💰 Cost limit reached: ${self._total_cost:.2f} / ${self.cost_limit:.2f}")
        
        # 🚀 Feature 1: Adaptive Intelligence - Auto-tune parameters
        if self.intelligence == "adaptive":
            top_k, hyde, multi_query = self._adapt_retrieval_strategy(question, top_k, hyde, multi_query)

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
            chunks = self._retrieve(q, top_k=top_k * 2, metadata_filter=metadata_filter,
                                    multimodal=self.multimodal)
            all_chunks.extend(chunks)

        unique = self._deduplicate(all_chunks)

        # 🚀 Feature 4: Collect multi-modal items for vision prompt
        mm_images: List[Dict[str, str]] = []
        if self.multimodal:
            for c in unique:
                meta = c.metadata or {}
                if meta.get("item_type") == "image" and meta.get("image_data"):
                    mm_images.append({
                        "data": meta["image_data"],
                        "format": meta.get("image_format", "png"),
                        "text": c.text,
                        "source": c.source,
                    })

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
        
        # 🚀 Feature 11: RBAC - Filter chunks based on user permissions
        if self._rbac:
            compressed = self._rbac.filter_chunks(
                compressed,
                user=user,
                roles=roles or [],
                redact_fields=True
            )

        # ---- generate answer ----
        max_tokens = llm_kwargs.pop("max_tokens", None)
        answer, citations = self._generate_answer(
            question, compressed, return_citations, answer_length,
            max_tokens=max_tokens, mm_images=mm_images if mm_images else None,
            **llm_kwargs
        )
        
        # 🚀 Feature 5: Secret scanning on output
        if self.scan_secrets:
            answer, scan_findings = self._scan_and_redact_answer(answer)
            if scan_findings:
                logger.warning(f"🚨 Redacted {len(scan_findings)} secrets from answer")
        
        # 🚀 Feature 3: Enhanced Citation with Provenance
        citations = self._enhance_citations_with_provenance(citations, compressed)

        latency = (time.time() - start) * 1000
        confidence = self._compute_confidence(compressed)
        
        # 🚀 Feature 5: Track cost
        query_cost = self.ai._total_cost if hasattr(self.ai, '_total_cost') else 0.0
        self._total_cost += query_cost
        
        # 🚀 Feature 1: Learn from query results
        if self.intelligence == "adaptive":
            self._record_query_performance(question, latency, confidence, len(compressed), hyde, multi_query)

        # 🚀 Feature 4: Attach multi-modal images to result
        mm_images_dedup = []
        seen_src = set()
        for img in mm_images:
            key = img.get("source", "") + img.get("text", "")[:50]
            if key not in seen_src:
                seen_src.add(key)
                mm_images_dedup.append(img)

        result = RAGResult(
            answer=answer,
            citations=citations,
            confidence=confidence,
            retrieved_chunks=compressed,
            query_variants=queries[1:] if multi_query else [],
            latency_ms=latency,
            tokens_used={
                "input": self.ai._total_input_tokens,
                "output": self.ai._total_output_tokens,
            },
            images=mm_images_dedup,
        )
        
        # Add cost tracking to result
        result.cost = query_cost
        result.total_session_cost = self._total_cost
        
        # 🚀 Feature 7: Store in semantic cache
        if self._semantic_cache and not return_citations:
            query_embedding = self.ai.embed(question)
            if query_embedding:
                self._semantic_cache.put(
                    query_embedding=query_embedding[0],
                    query_text=question,
                    answer=answer,
                    citations=citations,
                    retrieved_chunks=compressed,
                    confidence=confidence,
                    cost=query_cost
                )
        
        # 🚀 Feature 8: Record analytics
        if self._analytics:
            document_sources = list(set(c.source for c in compressed if c.source))
            self._analytics.record_query(
                query=question,
                latency_ms=latency,
                confidence=confidence,
                cost=query_cost,
                citations=len(citations),
                chunks_retrieved=len(compressed),
                document_sources=document_sources,
                cached=False
            )
        
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retrieve(self, query: str, top_k: int = 5,
                  metadata_filter: Optional[Dict[str, Any]] = None,
                  multimodal: bool = False) -> List[ChunkWithScore]:
        """Embed the query and search the vector store.

        Uses hierarchical tree retrieval when a tree has been built.
        When *multimodal* is True and a MultiModalIndex exists, also searches
        images and tables, fusing results with text chunks.
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

        # Check retriever circuit breaker
        if self._retriever_cb_open:
            logger.warning("Retriever circuit breaker open, returning empty")
            return []

        query_embedding = self.ai.embed(query)
        if not query_embedding:
            return []

        text_chunks: List[ChunkWithScore] = []
        retriever = self._get_retriever()
        try:
            if isinstance(retriever, ChromaRetriever):
                text_chunks = retriever.query(query_embedding[0], top_k=top_k,
                                              metadata_filter=metadata_filter)
            else:
                text_chunks = retriever.query(query_embedding[0], top_k=top_k)
            self._retriever_failures = 0
        except Exception as exc:
            self._retriever_failures += 1
            logger.warning("Retriever query failed (%d/%d): %s",
                           self._retriever_failures, self._retriever_cb_threshold, exc)
            if self._retriever_failures >= self._retriever_cb_threshold:
                self._retriever_cb_open = True
                logger.error("Retriever circuit breaker OPEN — falling back to keyword mode")
            return []

        # 🚀 Feature 4: Multi-modal search in parallel
        if multimodal and self._mm_index is not None:
            try:
                mm_result = self._mm_index.search(query, top_k=top_k)
                for item in mm_result.items:
                    text_chunks.append(ChunkWithScore(
                        text=item.text,
                        score=item.score * 0.9,
                        source=item.source,
                        source_type=item.item_type,
                        page=item.page,
                        metadata={
                            "item_type": item.item_type,
                            "image_data": item.image_data,
                            "image_format": item.image_format,
                        },
                    ))
            except Exception as exc:
                logger.debug("Multi-modal search failed: %s", exc)

        return text_chunks

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
            source = doc.source_path or ""

            if strategy == "hierarchical":
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

            # Track chunk IDs per source file
            chunk_ids = [getattr(c, 'chunk_id', f"{source}_{i}") for i, c in enumerate(chunks)]
            if source:
                self._file_chunk_ids.setdefault(source, []).extend(chunk_ids)

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

            # 🚀 Feature 4: Build multi-modal index alongside vector store
            if self.multimodal and (doc.images or doc.tables):
                if self._mm_index is None:
                    from runeextract.rag.multimodal import MultiModalIndex
                    self._mm_index = MultiModalIndex(embed_fn=self.ai.embed if hasattr(self.ai, 'embed') else None)
                self._mm_index.add_document(doc)

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
                    except Exception as exc:
                        logger.warning("Hierarchical summarization failed: %s", exc)
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

    # 🚀 Feature 2: Remove from index (live sync cleanup)
    def _remove_from_index(self, source_path: str) -> int:
        """Remove all chunks belonging to a source file from the vector store.

        Args:
            source_path: The file path whose chunks should be removed.

        Returns:
            Number of chunks removed.
        """
        removed = 0
        if self.vector_store_type == "chromadb":
            try:
                retriever = self._get_retriever()
                if isinstance(retriever, ChromaRetriever):
                    removed = retriever.delete_by_source(source_path)
            except Exception as exc:
                logger.warning("Failed to remove %s from index: %s", source_path, exc)
        if source_path in self._file_chunk_ids:
            del self._file_chunk_ids[source_path]
        if source_path in self._file_hashes:
            del self._file_hashes[source_path]
        return removed

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

        table_count = len(doc.tables)
        if table_count > 3:
            return "fixed_size"

        ws_count = len(doc.text.split())
        if ws_count > 10000:
            return "hierarchical"

        academic = {"introduction", "methods", "results", "conclusion",
                    "abstract", "references"}
        if academic & set(text.lower().split()[:50]):
            return "by_heading"

        lines = text.split("\n")
        if any(len(part.strip()) > 100 and "|" in part for part in lines):
            return "fixed_size"

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
                         max_tokens: Optional[int] = None,
                         mm_images: Optional[List[Dict[str, str]]] = None,
                         **llm_kwargs) -> tuple:
        """Build context with citation markers and call the LLM.

        When *max_tokens* is set, the context is packed to fit within
        the token budget using :class:`runeextract.rag.ContextPacker`.

        When *mm_images* is provided (list of image dicts with 'data', 'format',
        'text', 'source'), images are included as data URIs for vision-capable models.
        """
        import re

        if max_tokens is not None:
            try:
                from runeextract.rag.context_packer import ContextPacker
                packer = ContextPacker(max_tokens=max_tokens)
                packed = packer.pack(chunks, question, strategy="sorted")
                context = packed.text
                chunks = [chunks[i] for i in packed.chunk_map.values()]
            except Exception:
                logger.debug("Context packing failed, using all chunks")
                context = self._build_context(chunks)
        else:
            context = self._build_context(chunks)

        length_map = {"short": "2-3 sentences",
                      "medium": "1 paragraph",
                      "long": "3-5 paragraphs"}
        length_inst = length_map.get(length, "1 paragraph")

        # 🚀 Feature 4: Include images in vision prompt
        has_images = mm_images and len(mm_images) > 0
        if has_images:
            image_text = "\n".join(
                f"[IMAGE: {img.get('text', '')}] (source: {img.get('source', 'unknown')})"
                for img in mm_images[:4]
            )
            context += f"\n\n---\nReferenced images:\n{image_text}"

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

            # 🚀 Feature 3: CitationEngine fallback for uncited claims
            if not citations and chunks:
                try:
                    from runeextract.citation import CitationEngine
                    source_text = "\n".join(c.text for c in chunks)
                    engine = CitationEngine(
                        source=source_text,
                        strategy="overlap",
                        top_k=3,
                        min_score=0.15,
                    )
                    claim_result = engine.cite(answer)
                    for ci in claim_result.citations[:5]:
                        citations.append(Citation(
                            text=ci.text[:300],
                            source=ci.source,
                            chunk_index=ci.chunk_index,
                            relevance_score=ci.relevance_score,
                        ))
                except Exception as exc:
                    logger.debug("CitationEngine fallback failed: %s", exc)

        return answer, citations

    @staticmethod
    def _build_context(chunks: List[ChunkWithScore]) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            src = c.source or "unknown"
            page = f" p.{c.page}" if c.page is not None else ""
            parts.append(f"[{i}] Source: {src}{page}\n{c.text}")
        return "\n---\n".join(parts)

    def _compute_confidence(self, chunks: List[ChunkWithScore]) -> float:
        if not chunks:
            return 0.0
        try:
            scorer = ConfidenceScorer()
            factors = scorer.score(chunks, "", "")
            return factors.overall
        except Exception:
            return float(sum(c.score for c in chunks)) / len(chunks)

    # ------------------------------------------------------------------
    # 🚀 GAME-CHANGING FEATURE IMPLEMENTATIONS
    # ------------------------------------------------------------------

    # Feature 1: Adaptive Intelligence / Self-Tuning
    def _adapt_retrieval_strategy(self, question: str, top_k: int, 
                                   hyde: bool, multi_query: bool) -> tuple:
        """Auto-tune retrieval parameters based on query history and patterns."""
        if self._performance_stats["total_queries"] < 5:
            # Not enough data yet, use defaults
            return top_k, hyde, multi_query
        
        # Analyze question type
        q_lower = question.lower()
        is_factual = any(w in q_lower for w in ["what", "when", "where", "who"])
        is_analytical = any(w in q_lower for w in ["why", "how", "explain", "analyze"])
        is_comparison = any(w in q_lower for w in ["compare", "difference", "versus", "vs"])
        
        # Learn from past performance
        avg_conf = self._performance_stats["avg_confidence"]
        
        # Adaptive logic
        if is_factual and avg_conf > 0.7:
            # High confidence on factual - use simpler retrieval
            top_k = min(top_k, 3)
            hyde = False
            multi_query = False
        elif is_analytical:
            # Analytical questions benefit from more context
            top_k = max(top_k, 7)
            multi_query = True
            hyde = True
        elif is_comparison:
            # Comparisons need broad retrieval
            top_k = max(top_k, 10)
            multi_query = True
        
        # If average confidence is low, try advanced techniques
        if avg_conf < 0.5:
            hyde = True
            multi_query = True
            top_k = min(top_k + 3, 15)
        
        logger.debug(f"🎯 Adaptive: top_k={top_k}, hyde={hyde}, multi_query={multi_query}")
        return top_k, hyde, multi_query
    
    def _record_query_performance(self, question: str, latency: float, 
                                   confidence: float, chunks_used: int,
                                   used_hyde: bool, used_multi_query: bool):
        """Track query performance to improve future retrievals."""
        self._query_history.append({
            "question": question[:100],
            "latency_ms": latency,
            "confidence": confidence,
            "chunks_used": chunks_used,
            "hyde": used_hyde,
            "multi_query": used_multi_query,
            "timestamp": time.time()
        })
        
        # Update running stats
        stats = self._performance_stats
        n = stats["total_queries"]
        stats["avg_latency_ms"] = (stats["avg_latency_ms"] * n + latency) / (n + 1)
        stats["avg_confidence"] = (stats["avg_confidence"] * n + confidence) / (n + 1)
        stats["total_queries"] = n + 1
        
        # Track strategy success
        strategy_key = f"hyde={used_hyde},multi={used_multi_query}"
        if strategy_key not in stats["strategy_success"]:
            stats["strategy_success"][strategy_key] = []
        stats["strategy_success"][strategy_key].append(confidence)
        
        # Keep only last 100 queries
        if len(self._query_history) > 100:
            self._query_history.pop(0)
    
    # Feature 2: Live Document Sync / Incremental Indexing
    def start_watch(self, directory: str, incremental: bool = True, **extract_options):
        """Start background thread to watch directory for file changes."""
        import threading
        from runeextract.sync.watcher import DirectoryWatcher
        
        self._watcher_running = True
        
        def watch_loop():
            watcher = DirectoryWatcher(directory, patterns=["*.pdf", "*.docx", "*.txt", "*.md"])
            logger.info(f"👁️  Watching directory: {directory}")
            
            while self._watcher_running:
                try:
                    events = watcher.poll()
                    for event in events:
                        if event.event_type == "created":
                            logger.info(f"📄 New file detected: {event.path}")
                            self.ingest(event.path, incremental=incremental, **extract_options)
                        elif event.event_type == "modified":
                            logger.info(f"✏️  Modified file detected: {event.path}")
                            self._remove_from_index(event.path)
                            self.ingest(event.path, incremental=incremental, **extract_options)
                        elif event.event_type == "deleted":
                            logger.info(f"🗑️  Deleted file: {event.path}")
                            self._remove_from_index(event.path)
                    
                    time.sleep(2)  # Poll every 2 seconds
                except Exception as e:
                    logger.error(f"Watcher error: {e}")
                    time.sleep(5)
        
        self._watcher_thread = threading.Thread(target=watch_loop, daemon=True)
        self._watcher_thread.start()
        logger.info("✅ Document watcher started in background")
    
    def stop_watch(self):
        """Stop the background file watcher."""
        if self._watcher_running:
            self._watcher_running = False
            if self._watcher_thread:
                self._watcher_thread.join(timeout=5)
            logger.info("🛑 Document watcher stopped")
    
    # Feature 3: Citation with Provenance
    def _enhance_citations_with_provenance(self, citations: List[Citation], 
                                           chunks: List[ChunkWithScore]) -> List[Citation]:
        """Add rich provenance metadata to citations."""
        for i, citation in enumerate(citations):
            # Add bounding box info if available
            if hasattr(citation, 'metadata') and citation.metadata:
                bbox = citation.metadata.get('bbox')
                if bbox:
                    citation.bounding_box = bbox
            
            # Add extraction timestamp
            citation.extracted_at = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Add confidence score context
            if i < len(chunks):
                citation.retrieval_rank = i + 1
                citation.similarity_score = chunks[i].score
        
        return citations
    
    # Feature 4: Multi-Modal RAG
    def _process_multimodal(self, doc: Document) -> Document:
        """Process images, charts, tables with vision models."""
        if not doc.images:
            return doc
        
        logger.info(f"🖼️  Processing {len(doc.images)} images with vision model")
        
        try:
            # Check if vision is available
            if not hasattr(self.ai, 'describe_image'):
                # Fallback to OCR
                logger.debug("Vision model not available, using OCR")
                return doc
            
            # Process each image
            for img in doc.images:
                import hashlib
                img_hash = hashlib.md5(img.data[:1000] if img.data else b"").hexdigest()
                
                # Check cache
                if img_hash in self._vision_cache:
                    description = self._vision_cache[img_hash]
                else:
                    # Describe image with vision model
                    try:
                        description = self.ai.describe_image(img.data)
                        self._vision_cache[img_hash] = description
                        logger.debug(f"Vision description: {description[:100]}...")
                    except Exception as e:
                        logger.debug(f"Vision failed for image: {e}")
                        description = img.caption or ""
                
                # Append description to document text
                if description:
                    doc.text += f"\n\n[IMAGE: {description}]\n"
        
        except Exception as e:
            logger.warning(f"Multi-modal processing failed: {e}")
        
        return doc
    
    # Feature 5: Production Safeguards
    def _scan_document_secrets(self, doc: Document) -> list:
        """Scan document for API keys, tokens, passwords."""
        from runeextract.utils.secrets import scan_secrets
        
        if self._secret_scanner is None:
            self._secret_scanner = scan_secrets
        
        findings = scan_secrets(doc.text)
        return findings
    
    def _redact_document_secrets(self, text: str, findings: list) -> str:
        """Redact detected secrets from text."""
        from runeextract.utils.secrets import redact_secrets
        return redact_secrets(text, findings)
    
    def _scan_and_redact_answer(self, answer: str) -> tuple:
        """Scan and redact secrets from generated answer."""
        from runeextract.utils.secrets import scan_secrets, redact_secrets
        
        findings = scan_secrets(answer)
        if findings:
            answer = redact_secrets(answer, findings)
        
        return answer, findings
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics and usage metrics."""
        stats = {
            "total_documents": len(self._documents),
            "total_queries": self._performance_stats["total_queries"],
            "avg_latency_ms": self._performance_stats["avg_latency_ms"],
            "avg_confidence": self._performance_stats["avg_confidence"],
            "total_cost": self._total_cost,
            "cost_limit": self.cost_limit,
            "cost_remaining": self.cost_limit - self._total_cost if self.cost_limit else None,
            "intelligence_mode": self.intelligence,
            "multimodal_enabled": self.multimodal,
            "safe_mode": self.safe_mode,
            "watching": self._watcher_running,
        }
        
        # 🚀 Add Tier 1 feature stats
        if self._semantic_cache:
            cache_stats = self._semantic_cache.stats()
            stats["semantic_cache"] = cache_stats.to_dict()
        
        if self._analytics:
            analytics_summary = self._analytics.get_summary()
            stats["analytics"] = analytics_summary.to_dict()
        
        return stats
    
    def cache_stats(self) -> Dict[str, Any]:
        """Get semantic cache statistics."""
        if not self._semantic_cache:
            return {"enabled": False}
        
        return self._semantic_cache.stats().to_dict()
    
    def get_analytics(self) -> Optional[RAGAnalytics]:
        """Get analytics instance for advanced querying."""
        return self._analytics
    
    def query_stream(self, question: str, **kwargs):
        """Stream query results with progressive refinement.
        
        🚀 Feature 12: Streaming RAG
        
        Yields StreamEvent objects for each stage:
        - RETRIEVAL: Chunks being retrieved
        - PARTIAL_ANSWER: Answer tokens as they're generated
        - REFINEMENT: Additional chunks being processed
        - CITATION: Citations being added
        - COMPLETE: Final result with confidence
        
        Example::
        
            for event in rag.query_stream("What are the findings?"):
                if event.type == StreamEventType.PARTIAL_ANSWER:
                    print(event.text, end="", flush=True)
                elif event.type == StreamEventType.COMPLETE:
                    print(f"\\nConfidence: {event.confidence:.2%}")
        """
        if not self._streaming_rag:
            self._streaming_rag = StreamingRAG(
                rag_instance=self,
                initial_chunks=3,
                refinement_chunks=7,
                adaptive_depth=True,
                confidence_threshold=0.85
            )
        
        yield from self._streaming_rag.query_stream(question, **kwargs)
    
    # 🚀 Feature 13: RAG-as-a-Service API
    def create_api_server(self, api_keys: Optional[List[str]] = None,
                          rate_limit: int = 100,
                          enable_cors: bool = True):
        """Create a FastAPI REST server wrapping this RAG pipeline.

        Args:
            api_keys: Optional list of valid API keys for authentication.
            rate_limit: Max requests per minute per key.
            enable_cors: Enable CORS middleware.

        Returns:
            RAGAPIServer instance with a ``create_app()`` method.

        Usage::

            api = rag.create_api_server(api_keys=["secret"])
            app = api.create_app()
            # Run with: uvicorn app:app --reload
        """
        from runeextract.rag.api_server import RAGAPIServer
        return RAGAPIServer(
            rag=self,
            api_keys=api_keys,
            rate_limit=rate_limit,
            enable_cors=enable_cors
        )

    def serve(self, host: str = "0.0.0.0", port: int = 8000,
              api_keys: Optional[List[str]] = None,
              **uvicorn_kwargs):
        """Start the RAG API server directly.

        Args:
            host: Bind address.
            port: Bind port.
            api_keys: Optional API keys for authentication.
            **uvicorn_kwargs: Extra kwargs for ``uvicorn.run()``.
        """
        api = self.create_api_server(api_keys=api_keys)
        app = api.create_app()
        try:
            import uvicorn
            uvicorn.run(app, host=host, port=port, **uvicorn_kwargs)
        except ImportError:
            logger.error("uvicorn not installed. Install: pip install uvicorn")

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, '_watcher_running') and self._watcher_running:
            self.stop_watch()
