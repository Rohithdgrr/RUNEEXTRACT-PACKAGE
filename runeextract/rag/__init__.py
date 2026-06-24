"""
RuneExtract RAG — production RAG pipeline with zero-config setup.
"""

from runeextract.rag.auto_pipeline import AutoRAG, auto_rag
from runeextract.rag.types import RAGResult, Citation, ChunkWithScore
from runeextract.rag.retriever import ChromaRetriever, FAISSRetriever
from runeextract.rag.compressor import ContextualCompressor
from runeextract.rag.evaluate import RAGEvaluator
from runeextract.rag.hierarchical import HierarchicalChunker, SummaryNode, HierarchicalResult
from runeextract.rag.multimodal import MultiModalIndex, MultiModalItem, MultiModalResult
from runeextract.rag.instant import instant_rag
from runeextract.rag.templates import DomainConfig, DomainTemplates
from runeextract.rag.cache import RAGCache
from runeextract.rag.embedding_selector import resolve_embedding, get_domain_embedding
from runeextract.rag.query_router import QueryRouter, QueryIntent, DecomposedQuery
from runeextract.rag.hybrid_search import HybridSearch, HybridResult, BM25Sparse
from runeextract.rag.context_packer import ContextPacker, PackedContext
from runeextract.rag.robust_rag import RobustRAG, FallbackStrategy
from runeextract.rag.confidence import ConfidenceScorer, ConfidenceFactors
from runeextract.rag.debugger import RAGDebugger, DebugTrace
# 🚀 Feature 6: Smart Query Routing (v2 — multi-RAG orchestrator)
from runeextract.rag.routing import QueryRouter as SmartQueryRouter, RouteDecision
# 🚀 Feature 7: Semantic Caching
from runeextract.rag.semantic_cache import SemanticCache, CacheStats, CacheEntry
# 🚀 Feature 8: Analytics Dashboard
from runeextract.rag.analytics import RAGAnalytics, AnalyticsSummary, QueryMetrics
# 🚀 Feature 9: A/B Experiments
from runeextract.rag.experiments import (
    ExperimentManager, VariantConfig, VariantMetrics, ExperimentReport
)
# 🚀 Feature 10: Multi-Language
from runeextract.rag.multilingual import MultilingualRAG, TranslationCache
# 🚀 Feature 11: RBAC
from runeextract.rag.rbac import RBACManager, AccessRule, AuditLog
# 🚀 Feature 12: Streaming RAG
from runeextract.rag.streaming import StreamingRAG, StreamEvent, StreamEventType
# 🚀 Feature 13: RAG-as-a-Service API
from runeextract.rag.api_server import RAGAPIServer
# 🚀 Feature 14: Chain-of-Thought Reasoning
from runeextract.rag.reasoning import ChainOfThoughtReasoner, ReasoningStep, ReasoningTrace

__all__ = [
    "AutoRAG",
    "auto_rag",
    "instant_rag",
    "DomainConfig",
    "DomainTemplates",
    "RAGCache",
    "resolve_embedding",
    "get_domain_embedding",
    "QueryRouter",
    "QueryIntent",
    "DecomposedQuery",
    "HybridSearch",
    "HybridResult",
    "BM25Sparse",
    "ContextPacker",
    "PackedContext",
    "RobustRAG",
    "FallbackStrategy",
    "ConfidenceScorer",
    "ConfidenceFactors",
    "RAGDebugger",
    "DebugTrace",
    "RAGResult",
    "Citation",
    "ChunkWithScore",
    "ChromaRetriever",
    "FAISSRetriever",
    "ContextualCompressor",
    "RAGEvaluator",
    "HierarchicalChunker",
    "SummaryNode",
    "HierarchicalResult",
    "MultiModalIndex",
    "MultiModalItem",
    "MultiModalResult",
    # 🚀 Feature 6
    "SmartQueryRouter",
    "RouteDecision",
    # 🚀 Feature 7
    "SemanticCache",
    "CacheStats",
    "CacheEntry",
    # 🚀 Feature 8
    "RAGAnalytics",
    "AnalyticsSummary",
    "QueryMetrics",
    # 🚀 Feature 9
    "ExperimentManager",
    "VariantConfig",
    "VariantMetrics",
    "ExperimentReport",
    # 🚀 Feature 10
    "MultilingualRAG",
    "TranslationCache",
    # 🚀 Feature 11
    "RBACManager",
    "AccessRule",
    "AuditLog",
    # 🚀 Feature 12
    "StreamingRAG",
    "StreamEvent",
    "StreamEventType",
    # 🚀 Feature 13
    "RAGAPIServer",
    # 🚀 Feature 14
    "ChainOfThoughtReasoner",
    "ReasoningStep",
    "ReasoningTrace",
]
