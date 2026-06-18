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

__all__ = [
    "AutoRAG",
    "auto_rag",
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
]
