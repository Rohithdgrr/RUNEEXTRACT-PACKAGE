"""
RuneExtract - One extraction API for every document type.
"""

import asyncio

from runeextract.config import get_config, set_config
from runeextract.exceptions import ExtractionError, PathTraversalError, ExtractionTimeoutError, WrongPasswordError
from runeextract.models.document import Document, ChunkingStrategy, ChatSession

__version__ = "0.8.0"
__all__ = [
    # Core extraction
    "extract", "extract_many", "extract_many_with_errors",
    "extract_from_bytes", "extract_from_string",
    "extract_async", "extract_many_async",
    "extract_stream",
    "extract_crawl",
    # Core models
    "Document", "ChunkingStrategy", "ChatSession",
    "SecretFinding",
    # Configuration
    "get_config", "set_config",
    # Exceptions
    "ExtractionError", "PathTraversalError", "ExtractionTimeoutError", "WrongPasswordError",
    # RAG pipeline
    "AutoRAG", "auto_rag", "instant_rag", "RobustRAG",
    "QueryRouter", "SmartQueryRouter", "RouteDecision",
    # Security
    "scan_secrets", "redact_secrets",
    "DifferentialPrivacyEngine",
    # Structured extraction
    "StructuredExtractor", "extract_structured", "StructuredExtractionError",
    # Web
    "SmartCrawler", "CrawlResult", "smart_crawl",
    # Pipeline / orchestration
    "Pipeline", "PipelineStep", "PipelineContext", "PipelineResult", "run_pipeline",
    "DagPipeline", "ConditionalStep", "ParallelStep", "WaitStep",
    # Diff
    "DiffChange", "DiffResult", "DocumentComparator", "diff_documents",
    # Layout
    "LayoutElement", "BoundingBox", "LayoutParser",
    # Storage
    "StorageConnector", "S3Connector", "GCSConnector", "AzureConnector",
    # Graph
    "GraphNode", "GraphEdge", "DocumentGraph", "GraphBuilder",
    # Export
    "extract_from_presigned_url",
    "ONNXEmbeddingModel",
    "MinHashDeduplicator", "LSHDeduplicator", "EmbeddingDeduplicator",
    "deduplicate_embeddings",
    "KnowledgeBase",
    # Versioning & Budget & Dedup
    "IndexVersioning", "BudgetManager", "BudgetConfig",
    "BudgetExceededError", "DedupEngine", "DedupReport",
    # Signing
    "DocumentSigner", "DocumentVerifier", "SignatureInfo",
    "sign_document", "verify_document", "compute_document_hash",
    # Server
    "ExtractionServer", "run_server", "start_extraction_server",
    "WebSocketHandler",
    # Quality
    "QualityLevel", "QualityConfig", "extract_with_quality",
    "FastMode", "configure_quality",
    # Export
    "to_ragas", "to_dspy", "to_haystack", "to_llama_index",
    "to_jsonl", "ExportFormat",
    # RAG Fine-Tuning
    "FineTuneExample", "FineTuneDataset",
    "generate_fine_tuning_data",
    # Audit Features
    "PersistentEmbeddingCache", "IndexPersister", "IndexState",
    "TieredIndex", "TierConfig",
    "MultiTenantRAG", "TenantStore",
    "GraphRAGQuery",
    "RAGEvalSuite", "EvalQuestion", "EvalResult", "Scorecard",
    "MiddlewarePipeline", "RAGMiddleware", "LoggingMiddleware",
    "TimingMiddleware", "CacheMiddleware", "FallbackMiddleware",
    "apply_middleware",
]


class _Pipeline:
    """Proxy that supports both Pipeline(steps...) and Pipeline.extract()."""
    _cls = None

    @classmethod
    def _get_cls(cls):
        if cls._cls is None:
            from runeextract.transform.pipeline import Pipeline as _P
            cls._cls = _P
        return cls._cls

    def __getattr__(self, name):
        return getattr(self._get_cls(), name)

    def __call__(self, *args, **kwargs):
        return self._get_cls()(*args, **kwargs)

    def __repr__(self):
        return repr(self._get_cls())


Pipeline = _Pipeline()


# ── Module-level lazy import dispatch ──────────────────────────────────────
# All public API items are loaded on first access via __getattr__.
# This replaces ~95 hand-written lazy wrapper functions with a single dispatch.

_LAZY_IMPORTS = {
    # Core extraction
    "extract": "runeextract.core.extraction:extract",
    "extract_from_bytes": "runeextract.core.extraction:extract_from_bytes",
    "extract_from_string": "runeextract.core.extraction:extract_from_string",
    "extract_many": "runeextract.core.extraction:extract_many",
    "extract_many_with_errors": "runeextract.core.extraction:extract_many_with_errors",
    "extract_async": "runeextract.core.extraction:extract_async",
    "extract_many_async": "runeextract.core.extraction:extract_many_async",
    "extract_stream": "runeextract.core.extraction:extract_stream",
    "extract_and_index": "runeextract.core.extraction:extract_and_index",
    "extract_crawl": "runeextract.core.extraction:extract_crawl",
    # Async extractor
    "extract_async_url": "runeextract.core.async_extractor:extract_async_url",
    "extract_many_async_url": "runeextract.core.async_extractor:extract_many_async_url",
    "batch_process": "runeextract.core.async_extractor:batch_process",
    "ProcessPoolExtractor": "runeextract.core.async_extractor:ProcessPoolExtractor",
    "cleanup": "runeextract.core.async_extractor:cleanup",
    # RAG
    "AutoRAG": "runeextract.rag.auto_pipeline:AutoRAG",
    "auto_rag": "runeextract.rag.auto_pipeline:auto_rag",
    "instant_rag": "runeextract.rag.instant:instant_rag",
    "RobustRAG": "runeextract.rag.robust_rag:RobustRAG",
    "KnowledgeBase": "runeextract.rag.knowledge_base:KnowledgeBase",
    "IndexVersioning": "runeextract.rag.versioning:IndexVersioning",
    "BudgetManager": "runeextract.rag.budget:BudgetManager",
    "BudgetConfig": "runeextract.rag.budget:BudgetConfig",
    "BudgetExceededError": "runeextract.rag.budget:BudgetExceededError",
    "DedupEngine": "runeextract.rag.dedup_engine:DedupEngine",
    "DedupReport": "runeextract.rag.dedup_engine:DedupReport",
    "RAGDebugger": "runeextract.rag.debugger:RAGDebugger",
    "ConfidenceScorer": "runeextract.rag.confidence:ConfidenceScorer",
    "QueryRouter": "runeextract.rag.routing:QueryRouter",
    "SmartQueryRouter": "runeextract.rag.routing:SmartQueryRouter",
    "RouteDecision": "runeextract.rag.routing:RouteDecision",
    "QueryAnalyzer": "runeextract.rag.query_analyzer:QueryAnalyzer",
    # RAG audit features
    "PersistentEmbeddingCache": "runeextract.rag.embed_cache:PersistentEmbeddingCache",
    "IndexPersister": "runeextract.rag.persistence:IndexPersister",
    "IndexState": "runeextract.rag.persistence:IndexState",
    "TieredIndex": "runeextract.rag.tiered_index:TieredIndex",
    "TierConfig": "runeextract.rag.tiered_index:TierConfig",
    "MultiTenantRAG": "runeextract.rag.tenant:MultiTenantRAG",
    "TenantStore": "runeextract.rag.tenant:TenantStore",
    "GraphRAGQuery": "runeextract.rag.graph_rag:GraphRAGQuery",
    "RAGEvalSuite": "runeextract.rag.eval_suite:RAGEvalSuite",
    "EvalQuestion": "runeextract.rag.eval_suite:EvalQuestion",
    "EvalResult": "runeextract.rag.eval_suite:EvalResult",
    "Scorecard": "runeextract.rag.eval_suite:Scorecard",
    "MiddlewarePipeline": "runeextract.rag.middleware:MiddlewarePipeline",
    "RAGMiddleware": "runeextract.rag.middleware:RAGMiddleware",
    "LoggingMiddleware": "runeextract.rag.middleware:LoggingMiddleware",
    "TimingMiddleware": "runeextract.rag.middleware:TimingMiddleware",
    "CacheMiddleware": "runeextract.rag.middleware:CacheMiddleware",
    "FallbackMiddleware": "runeextract.rag.middleware:FallbackMiddleware",
    "apply_middleware": "runeextract.rag.middleware:apply_middleware",
    # Fine-tuning
    "FineTuneExample": "runeextract.rag.fine_tune:FineTuneExample",
    "FineTuneDataset": "runeextract.rag.fine_tune:FineTuneDataset",
    "generate_fine_tuning_data": "runeextract.rag.fine_tune:generate_fine_tuning_data",
    # Security
    "scan_secrets": "runeextract.utils.secrets:scan_secrets",
    "redact_secrets": "runeextract.utils.secrets:redact_secrets",
    "MemoryProfiler": "runeextract.utils.memory:MemoryProfiler",
    "DifferentialPrivacyEngine": "runeextract.utils.privacy:DifferentialPrivacyEngine",
    "SecretFinding": "runeextract.utils.secrets:SecretFinding",
    # Structured extraction
    "StructuredExtractor": "runeextract.structured.extractor:StructuredExtractor",
    "extract_structured": "runeextract.structured.extractor:extract_structured",
    "StructuredExtractionError": "runeextract.exceptions:StructuredExtractionError",
    # Citation
    "CitationEngine": "runeextract.citation.engine:CitationEngine",
    "CitationResult": "runeextract.citation.engine:CitationResult",
    "cite_document": "runeextract.citation.engine:cite_document",
    # Web
    "SmartCrawler": "runeextract.web.crawler:SmartCrawler",
    "CrawlResult": "runeextract.web.crawler:CrawlResult",
    "smart_crawl": "runeextract.web.crawler:smart_crawl",
    "parse_sitemap": "runeextract.web.sitemap:parse_sitemap",
    "discover_sitemap": "runeextract.web.sitemap:discover_sitemap",
    "parse_feed": "runeextract.web.feed:parse_feed",
    "discover_feed": "runeextract.web.feed:discover_feed",
    # Transform / Pipeline
    "PipelineStep": "runeextract.transform.pipeline:PipelineStep",
    "PipelineContext": "runeextract.transform.pipeline:PipelineContext",
    "PipelineResult": "runeextract.transform.pipeline:PipelineResult",
    "run_pipeline": "runeextract.transform.pipeline:run_pipeline",
    "DagPipeline": "runeextract.transform.pipeline:DagPipeline",
    "ConditionalStep": "runeextract.transform.pipeline:ConditionalStep",
    "ParallelStep": "runeextract.transform.pipeline:ParallelStep",
    "WaitStep": "runeextract.transform.pipeline:WaitStep",
    # Sync
    "DirectoryWatcher": "runeextract.sync.watcher:DirectoryWatcher",
    "FileEvent": "runeextract.sync.watcher:FileEvent",
    "FileSync": "runeextract.sync.syncer:FileSync",
    "poll_directory": "runeextract.sync.extractor:poll_directory",
    "sync_directories": "runeextract.sync.syncer:sync_directories",
    "scan_and_extract": "runeextract.sync.extractor:scan_and_extract",
    "watch_and_extract": "runeextract.sync.extractor:watch_and_extract",
    # Agent / MCP
    "mcp_tool_extract": "runeextract.agent.mcp_server:mcp_tool_extract",
    "mcp_tool_extract_many": "runeextract.agent.mcp_server:mcp_tool_extract_many",
    "mcp_tool_search": "runeextract.agent.mcp_server:mcp_tool_search",
    "mcp_tool_extract_url": "runeextract.agent.mcp_server:mcp_tool_extract_url",
    "mcp_tool_ask": "runeextract.agent.mcp_server:mcp_tool_ask",
    "mcp_tool_chunk": "runeextract.agent.mcp_server:mcp_tool_chunk",
    "run_mcp_server": "runeextract.agent.mcp_server:run_mcp_server",
    "RuneExtractLoader": "runeextract.agent.langchain:RuneExtractLoader",
    "RuneExtractTransformer": "runeextract.agent.langchain:RuneExtractTransformer",
    "RuneExtractReader": "runeextract.agent.llamaindex:RuneExtractReader",
    "RuneExtractTool": "runeextract.agent.crewai:RuneExtractTool",
    "autogen_extract_tool": "runeextract.agent.autogen:autogen_extract_tool",
    "RuneExtractGraphTool": "runeextract.agent.langgraph:RuneExtractGraphTool",
    "RuneExtractSearchTool": "runeextract.agent.langgraph:RuneExtractSearchTool",
    "RuneExtractAskTool": "runeextract.agent.langgraph:RuneExtractAskTool",
    "main_cli": "runeextract.cli.main:main_cli",
    # Layout
    "LayoutElement": "runeextract.layout.parser:LayoutElement",
    "BoundingBox": "runeextract.layout.parser:BoundingBox",
    "LayoutParser": "runeextract.layout.parser:LayoutParser",
    "parse_layout": "runeextract.layout.parser:parse_layout",
    "get_reading_order": "runeextract.layout.parser:get_reading_order",
    # Diff
    "DiffChange": "runeextract.diff.comparator:DiffChange",
    "DiffResult": "runeextract.diff.comparator:DiffResult",
    "DocumentComparator": "runeextract.diff.comparator:DocumentComparator",
    "diff_documents": "runeextract.diff.comparator:diff_documents",
    "compare_files": "runeextract.diff.comparator:compare_files",
    # ONNX embeddings
    "ONNXEmbeddingModel": "runeextract.embeddings.onnx:ONNXEmbeddingModel",
    "get_onnx_embedding": "runeextract.embeddings.onnx:get_onnx_embedding",
    # Storage
    "StorageConnector": "runeextract.storage.connectors:StorageConnector",
    "S3Connector": "runeextract.storage.connectors:S3Connector",
    "GCSConnector": "runeextract.storage.connectors:GCSConnector",
    "AzureConnector": "runeextract.storage.connectors:AzureConnector",
    "get_storage_connector": "runeextract.storage.connectors:get_storage_connector",
    "extract_from_presigned_url": "runeextract.storage.presigned:extract_from_presigned_url",
    # Graph
    "build_document_graph": "runeextract.graph.builder:build_document_graph",
    "query_graph": "runeextract.graph.builder:query_graph",
    "GraphNode": "runeextract.graph.builder:GraphNode",
    "GraphEdge": "runeextract.graph.builder:GraphEdge",
    "DocumentGraph": "runeextract.graph.builder:DocumentGraph",
    "GraphBuilder": "runeextract.graph.builder:GraphBuilder",
    # Vision
    "VisionAnalyzer": "runeextract.vision.analyzer:VisionAnalyzer",
    "ChartInterpretation": "runeextract.vision.analyzer:ChartInterpretation",
    "FigureCaption": "runeextract.vision.analyzer:FigureCaption",
    "describe_image": "runeextract.vision.analyzer:describe_image",
    "interpret_chart": "runeextract.vision.analyzer:interpret_chart",
    "caption_figure": "runeextract.vision.analyzer:caption_figure",
    # OCR
    "OCRLanguageDetector": "runeextract.ocr:OCRLanguageDetector",
    "detect_ocr_language": "runeextract.ocr:detect_ocr_language",
    "get_tesseract_langs": "runeextract.ocr:get_tesseract_langs",
    "get_ocr_languages": "runeextract.ocr:get_ocr_languages",
    # TOC
    "TOCEntry": "runeextract.toc:TOCEntry",
    "TOCParser": "runeextract.toc:TOCParser",
    "extract_toc": "runeextract.toc:extract_toc",
    "toc_to_markdown": "runeextract.toc:toc_to_markdown",
    "toc_to_dict": "runeextract.toc:toc_to_dict",
    "toc_to_json": "runeextract.toc:toc_to_json",
    # Dedup
    "MinHashDeduplicator": "runeextract.dedup.minhash:MinHashDeduplicator",
    "LSHDeduplicator": "runeextract.dedup.minhash:LSHDeduplicator",
    "EmbeddingDeduplicator": "runeextract.dedup.minhash:EmbeddingDeduplicator",
    "deduplicate": "runeextract.dedup.minhash:deduplicate",
    "deduplicate_documents": "runeextract.dedup.minhash:deduplicate_documents",
    "deduplicate_embeddings": "runeextract.dedup.minhash:deduplicate_embeddings",
    # Signing
    "DocumentSigner": "runeextract.signing.signer:DocumentSigner",
    "DocumentVerifier": "runeextract.signing.signer:DocumentVerifier",
    "SignatureInfo": "runeextract.signing.signer:SignatureInfo",
    "sign_document": "runeextract.signing.signer:sign_document",
    "verify_document": "runeextract.signing.signer:verify_document",
    "compute_document_hash": "runeextract.signing.signer:compute_document_hash",
    "generate_signing_keypair": "runeextract.signing.signer:generate_signing_keypair",
    # Server
    "ExtractionServer": "runeextract.server.server:ExtractionServer",
    "run_server": "runeextract.server.server:run_server",
    "start_extraction_server": "runeextract.server.server:start_extraction_server",
    "WebSocketHandler": "runeextract.server.server:WebSocketHandler",
    # Quality
    "QualityLevel": "runeextract.quality:QualityLevel",
    "QualityConfig": "runeextract.quality:QualityConfig",
    "extract_with_quality": "runeextract.quality:extract_with_quality",
    "FastMode": "runeextract.quality:FastMode",
    "configure_quality": "runeextract.quality:configure_quality",
    # Export formats
    "to_ragas": "runeextract.export:to_ragas",
    "to_dspy": "runeextract.export:to_dspy",
    "to_haystack": "runeextract.export:to_haystack",
    "to_llama_index": "runeextract.export:to_llama_index",
    "to_jsonl": "runeextract.export:to_jsonl",
    "ExportFormat": "runeextract.export:ExportFormat",
    # Benchmarks
    "BenchmarkRunner": "runeextract.benchmarks:BenchmarkRunner",
    "run_all_benchmarks": "runeextract.benchmarks:run_all_benchmarks",
    # CLI
    "doctor": "runeextract.cli.doctor:doctor",
    "eval_cli": "runeextract.cli.eval_cli:eval_cli",
}


def __getattr__(name):
    """Lazy-import any name listed in _LAZY_IMPORTS on first access."""
    if name in _LAZY_IMPORTS:
        import importlib
        path = _LAZY_IMPORTS[name]
        module_path, _, attr_name = path.partition(":")
        module = importlib.import_module(module_path)
        attr = getattr(module, attr_name)
        setattr(__import__(__name__), name, attr)
        return attr
    raise AttributeError(f"module 'runeextract' has no attribute '{name}'")
