"""
RuneExtract - One extraction API for every document type.
"""

import asyncio

from runeextract.config import get_config, set_config
from runeextract.exceptions import ExtractionError, PathTraversalError, ExtractionTimeoutError, WrongPasswordError
from runeextract.models.document import Document, ChunkingStrategy, ChatSession

__version__ = "0.7.0"
__all__ = [
    "extract", "extract_many", "extract_many_with_errors",
    "extract_async", "extract_many_async", "extract_async_url", "extract_many_async_url", "batch_process", "ProcessPoolExtractor", "extract_and_index",
    "extract_stream", "extract_from_bytes", "extract_from_string",
    "extract_crawl",
    "Document", "ChunkingStrategy", "get_config", "set_config",
    "AutoRAG", "auto_rag", "instant_rag", "RobustRAG", "RAGDebugger", "ConfidenceScorer", "QueryRouter",
    "scan_secrets", "redact_secrets", "MemoryProfiler",
    "DifferentialPrivacyEngine", "SecretFinding", "WrongPasswordError",
    "StructuredExtractor", "extract_structured", "StructuredExtractionError",
    "CitationEngine", "CitationResult", "cite_document",
    "SmartCrawler", "CrawlResult", "smart_crawl",
    "parse_sitemap", "discover_sitemap",
    "parse_feed", "discover_feed",
    "Pipeline", "PipelineStep", "PipelineContext", "PipelineResult", "run_pipeline",
    "DagPipeline", "ConditionalStep", "ParallelStep", "WaitStep",
    "DirectoryWatcher", "FileEvent", "poll_directory",
    "FileSync", "sync_directories",
    "scan_and_extract", "watch_and_extract",
    "mcp_tool_extract", "mcp_tool_extract_many", "mcp_tool_search",
    "RuneExtractLoader", "RuneExtractTransformer",
    "RuneExtractReader",
    "RuneExtractTool",
    "autogen_extract_tool",
    "LayoutElement", "BoundingBox", "LayoutParser",
    "parse_layout", "get_reading_order",
    "DiffChange", "DiffResult", "DocumentComparator",
    "diff_documents", "compare_files",
    "ONNXEmbeddingModel", "get_onnx_embedding",
    "StorageConnector", "S3Connector", "GCSConnector", "AzureConnector", "get_storage_connector",
    "MinHashDeduplicator", "LSHDeduplicator", "EmbeddingDeduplicator",
    "deduplicate", "deduplicate_documents",
    "ExtractionServer",
    "VisionAnalyzer", "ChartInterpretation", "FigureCaption",
    "describe_image", "interpret_chart", "caption_figure",
    "GraphNode", "GraphEdge", "DocumentGraph", "GraphBuilder",
    "build_document_graph", "query_graph",
    "extract_from_presigned_url",
    "TOCEntry", "TOCParser", "extract_toc", "toc_to_markdown", "toc_to_dict", "toc_to_json",
    "OCRLanguageDetector", "detect_ocr_language", "get_tesseract_langs", "get_ocr_languages",
    "FastMode", "QualityLevel", "configure_quality",
    "cleanup", "ChatSession",
]

from runeextract.core.extraction import (
    extract, extract_from_bytes, extract_from_string,
    extract_many, extract_many_with_errors,
    extract_async, extract_many_async, extract_stream, extract_and_index,
    extract_crawl,
)

# --- True Async Extractor (requires async extra) ---


async def extract_async_url(url: str, **kwargs):
    from runeextract.core.async_extractor import extract_async_url as _e
    return await _e(url, **kwargs)


async def extract_many_async_url(urls: list, **kwargs):
    from runeextract.core.async_extractor import extract_many_async_url as _e
    return await _e(urls, **kwargs)


async def batch_process(items: list, fn, **kwargs):
    from runeextract.core.async_extractor import batch_process as _b
    return await _b(items, fn, **kwargs)


def ProcessPoolExtractor(**kwargs):
    from runeextract.core.async_extractor import ProcessPoolExtractor as _P
    return _P(**kwargs)


async def cleanup():
    from runeextract.core.async_extractor import cleanup as _c
    return await _c()




def AutoRAG(*args, **kwargs):
    """Lazy import for :class:`runeextract.rag.AutoRAG`."""
    from runeextract.rag.auto_pipeline import AutoRAG as _AutoRAG
    return _AutoRAG(*args, **kwargs)


def auto_rag(*args, **kwargs):
    """Lazy import for :func:`runeextract.rag.auto_rag`."""
    from runeextract.rag.auto_pipeline import auto_rag as _auto_rag
    return _auto_rag(*args, **kwargs)


def instant_rag(*args, **kwargs):
    """Lazy import for :func:`runeextract.rag.instant_rag`."""
    from runeextract.rag.instant import instant_rag as _instant_rag
    return _instant_rag(*args, **kwargs)


def RobustRAG(*args, **kwargs):
    """Lazy import for :class:`runeextract.rag.RobustRAG`."""
    from runeextract.rag.robust_rag import RobustRAG as _RR
    return _RR(*args, **kwargs)


def RAGDebugger(*args, **kwargs):
    """Lazy import for :class:`runeextract.rag.RAGDebugger`."""
    from runeextract.rag.debugger import RAGDebugger as _RD
    return _RD(*args, **kwargs)


def ConfidenceScorer(*args, **kwargs):
    """Lazy import for :class:`runeextract.rag.ConfidenceScorer`."""
    from runeextract.rag.confidence import ConfidenceScorer as _CS
    return _CS(*args, **kwargs)


def QueryRouter(*args, **kwargs):
    """Lazy import for :class:`runeextract.rag.QueryRouter`."""
    from runeextract.rag.query_router import QueryRouter as _QR
    return _QR(*args, **kwargs)


# --- Tier 3 Security: lazy imports ---


def scan_secrets(text: str) -> list:
    """Scan text for API keys, tokens, passwords, and other secrets."""
    from runeextract.utils.secrets import scan_secrets as _scan
    return _scan(text)


def redact_secrets(text: str, findings: list) -> str:
    """Redact detected secrets from text using finding positions."""
    from runeextract.utils.secrets import redact_secrets as _redact
    return _redact(text, findings)


def MemoryProfiler(warn_mb: float = 500.0, limit_mb: float = 0.0, enabled: bool = True):
    """Create a MemoryProfiler for profiling extraction memory usage."""
    from runeextract.utils.memory import MemoryProfiler as _MP
    return _MP(warn_mb=warn_mb, limit_mb=limit_mb, enabled=enabled)


def DifferentialPrivacyEngine(epsilon: float = 1.0, delta: float = 0.0):
    """Create a DifferentialPrivacyEngine for private PII redaction."""
    from runeextract.utils.privacy import DifferentialPrivacyEngine as _DP
    return _DP(epsilon=epsilon, delta=delta)


def SecretFinding(*args, **kwargs):
    """Lazy import for SecretFinding dataclass."""
    from runeextract.utils.secrets import SecretFinding as _SF
    return _SF(*args, **kwargs)


# --- Structured Extraction ---


def StructuredExtractor(*args, **kwargs):
    """Lazy import for :class:`runeextract.structured.StructuredExtractor`."""
    from runeextract.structured.extractor import StructuredExtractor as _SE
    return _SE(*args, **kwargs)


def extract_structured(*args, **kwargs):
    """Lazy import for :func:`runeextract.structured.extract_structured`."""
    from runeextract.structured.extractor import extract_structured as _es
    return _es(*args, **kwargs)


# --- Citation Engine ---


def CitationEngine(*args, **kwargs):
    """Lazy import for :class:`runeextract.citation.CitationEngine`."""
    from runeextract.citation.engine import CitationEngine as _CE
    return _CE(*args, **kwargs)


def CitationResult(*args, **kwargs):
    """Lazy import for :class:`runeextract.citation.CitationResult`."""
    from runeextract.citation.engine import CitationResult as _CR
    return _CR(*args, **kwargs)


def cite_document(*args, **kwargs):
    """Lazy import for :func:`runeextract.citation.cite_document`."""
    from runeextract.citation.engine import cite_document as _cd
    return _cd(*args, **kwargs)


# --- Web / Crawler ---


def SmartCrawler(*args, **kwargs):
    """Lazy import for :class:`runeextract.web.SmartCrawler`."""
    from runeextract.web.crawler import SmartCrawler as _SC
    return _SC(*args, **kwargs)


def CrawlResult(*args, **kwargs):
    """Lazy import for :class:`runeextract.web.CrawlResult`."""
    from runeextract.web.crawler import CrawlResult as _CR
    return _CR(*args, **kwargs)


def smart_crawl(*args, **kwargs):
    """Lazy import for :func:`runeextract.web.smart_crawl`."""
    from runeextract.web.crawler import smart_crawl as _sc
    return _sc(*args, **kwargs)


def parse_sitemap(*args, **kwargs):
    """Lazy import for :func:`runeextract.web.parse_sitemap`."""
    from runeextract.web.sitemap import parse_sitemap as _ps
    return _ps(*args, **kwargs)


def discover_sitemap(*args, **kwargs):
    """Lazy import for :func:`runeextract.web.discover_sitemap`."""
    from runeextract.web.sitemap import discover_sitemap as _ds
    return _ds(*args, **kwargs)


def parse_feed(*args, **kwargs):
    """Lazy import for :func:`runeextract.web.parse_feed`."""
    from runeextract.web.feed import parse_feed as _pf
    return _pf(*args, **kwargs)


def discover_feed(*args, **kwargs):
    """Lazy import for :func:`runeextract.web.discover_feed`."""
    from runeextract.web.feed import discover_feed as _df
    return _df(*args, **kwargs)


# --- Transform / Pipeline ---


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


def PipelineStep(*args, **kwargs):
    """Lazy import for :class:`runeextract.transform.PipelineStep`."""
    from runeextract.transform.pipeline import PipelineStep as _PS
    return _PS(*args, **kwargs)


def PipelineContext(*args, **kwargs):
    """Lazy import for :class:`runeextract.transform.PipelineContext`."""
    from runeextract.transform.pipeline import PipelineContext as _PC
    return _PC(*args, **kwargs)


def PipelineResult(*args, **kwargs):
    """Lazy import for :class:`runeextract.transform.PipelineResult`."""
    from runeextract.transform.pipeline import PipelineResult as _PR
    return _PR(*args, **kwargs)


def run_pipeline(*args, **kwargs):
    """Lazy import for :func:`runeextract.transform.run_pipeline`."""
    from runeextract.transform.pipeline import run_pipeline as _rp
    return _rp(*args, **kwargs)


def DagPipeline(*args, **kwargs):
    """Lazy import for :class:`runeextract.transform.DagPipeline`."""
    from runeextract.transform.pipeline import DagPipeline as _D
    return _D(*args, **kwargs)


def ConditionalStep(*args, **kwargs):
    """Lazy import for :class:`runeextract.transform.ConditionalStep`."""
    from runeextract.transform.pipeline import ConditionalStep as _C
    return _C(*args, **kwargs)


def ParallelStep(*args, **kwargs):
    """Lazy import for :class:`runeextract.transform.ParallelStep`."""
    from runeextract.transform.pipeline import ParallelStep as _P
    return _P(*args, **kwargs)


def WaitStep(*args, **kwargs):
    """Lazy import for :class:`runeextract.transform.WaitStep`."""
    from runeextract.transform.pipeline import WaitStep as _W
    return _W(*args, **kwargs)


# --- File System Sync ---


def DirectoryWatcher(*args, **kwargs):
    """Lazy import for :class:`runeextract.sync.DirectoryWatcher`."""
    from runeextract.sync.watcher import DirectoryWatcher as _DW
    return _DW(*args, **kwargs)


def FileEvent(*args, **kwargs):
    """Lazy import for :class:`runeextract.sync.FileEvent`."""
    from runeextract.sync.watcher import FileEvent as _FE
    return _FE(*args, **kwargs)


def poll_directory(*args, **kwargs):
    """Lazy import for :func:`runeextract.sync.poll_directory`."""
    from runeextract.sync.watcher import poll_directory as _pd
    return _pd(*args, **kwargs)


def FileSync(*args, **kwargs):
    """Lazy import for :class:`runeextract.sync.FileSync`."""
    from runeextract.sync.syncer import FileSync as _FS
    return _FS(*args, **kwargs)


def sync_directories(*args, **kwargs):
    """Lazy import for :func:`runeextract.sync.sync_directories`."""
    from runeextract.sync.syncer import sync_directories as _sd
    return _sd(*args, **kwargs)


def scan_and_extract(*args, **kwargs):
    """Lazy import for :func:`runeextract.sync.scan_and_extract`."""
    from runeextract.sync.extractor import scan_and_extract as _se
    return _se(*args, **kwargs)


def watch_and_extract(*args, **kwargs):
    """Lazy import for :func:`runeextract.sync.watch_and_extract`."""
    from runeextract.sync.extractor import watch_and_extract as _we
    return _we(*args, **kwargs)


# --- Agent SDK Integrations ---


def mcp_tool_extract(*args, **kwargs):
    """Lazy import for :func:`runeextract.agent.mcp_tool_extract`."""
    from runeextract.agent.mcp_server import mcp_tool_extract as _mcp
    return _mcp(*args, **kwargs)


def mcp_tool_extract_many(*args, **kwargs):
    """Lazy import for :func:`runeextract.agent.mcp_tool_extract_many`."""
    from runeextract.agent.mcp_server import mcp_tool_extract_many as _mcp
    return _mcp(*args, **kwargs)


def mcp_tool_search(*args, **kwargs):
    """Lazy import for :func:`runeextract.agent.mcp_tool_search`."""
    from runeextract.agent.mcp_server import mcp_tool_search as _mcp
    return _mcp(*args, **kwargs)


def RuneExtractLoader(*args, **kwargs):
    """Lazy import for :class:`runeextract.agent.RuneExtractLoader`."""
    from runeextract.agent.langchain import RuneExtractLoader as _L
    return _L(*args, **kwargs)


def RuneExtractTransformer(*args, **kwargs):
    """Lazy import for :class:`runeextract.agent.RuneExtractTransformer`."""
    from runeextract.agent.langchain import RuneExtractTransformer as _T
    return _T(*args, **kwargs)


def RuneExtractReader(*args, **kwargs):
    """Lazy import for :class:`runeextract.agent.RuneExtractReader`."""
    from runeextract.agent.llamaindex import RuneExtractReader as _R
    return _R(*args, **kwargs)


def RuneExtractTool(*args, **kwargs):
    """Lazy import for :class:`runeextract.agent.RuneExtractTool`."""
    from runeextract.agent.crewai import RuneExtractTool as _T
    return _T(*args, **kwargs)


def autogen_extract_tool(*args, **kwargs):
    """Lazy import for :func:`runeextract.agent.autogen_extract_tool`."""
    from runeextract.agent.autogen import autogen_extract_tool as _at
    return _at(*args, **kwargs)


# --- Layout-aware parsing ---


def LayoutElement(*args, **kwargs):
    """Lazy import for :class:`runeextract.layout.LayoutElement`."""
    from runeextract.layout.parser import LayoutElement as _LE
    return _LE(*args, **kwargs)


def BoundingBox(*args, **kwargs):
    """Lazy import for :class:`runeextract.layout.BoundingBox`."""
    from runeextract.layout.parser import BoundingBox as _BB
    return _BB(*args, **kwargs)


def LayoutParser(*args, **kwargs):
    """Lazy import for :class:`runeextract.layout.LayoutParser`."""
    from runeextract.layout.parser import LayoutParser as _LP
    return _LP(*args, **kwargs)


def parse_layout(*args, **kwargs):
    """Lazy import for :func:`runeextract.layout.parse_layout`."""
    from runeextract.layout.parser import parse_layout as _pl
    return _pl(*args, **kwargs)


def get_reading_order(*args, **kwargs):
    """Lazy import for :func:`runeextract.layout.get_reading_order`."""
    from runeextract.layout.parser import get_reading_order as _gro
    return _gro(*args, **kwargs)


# --- Document diff / version tracking ---


def DiffChange(*args, **kwargs):
    """Lazy import for :class:`runeextract.diff.DiffChange`."""
    from runeextract.diff.comparator import DiffChange as _DC
    return _DC(*args, **kwargs)


def DiffResult(*args, **kwargs):
    """Lazy import for :class:`runeextract.diff.DiffResult`."""
    from runeextract.diff.comparator import DiffResult as _DR
    return _DR(*args, **kwargs)


def DocumentComparator(*args, **kwargs):
    """Lazy import for :class:`runeextract.diff.DocumentComparator`."""
    from runeextract.diff.comparator import DocumentComparator as _DC
    return _DC(*args, **kwargs)


def diff_documents(*args, **kwargs):
    """Lazy import for :func:`runeextract.diff.diff_documents`."""
    from runeextract.diff.comparator import diff_documents as _dd
    return _dd(*args, **kwargs)


def compare_files(*args, **kwargs):
    """Lazy import for :func:`runeextract.diff.compare_files`."""
    from runeextract.diff.comparator import compare_files as _cf
    return _cf(*args, **kwargs)


# --- ONNX Embeddings ---


def ONNXEmbeddingModel(*args, **kwargs):
    """Lazy import for :class:`runeextract.embeddings.ONNXEmbeddingModel`."""
    from runeextract.embeddings.onnx import ONNXEmbeddingModel as _O
    return _O(*args, **kwargs)


def get_onnx_embedding(*args, **kwargs):
    """Lazy import for :func:`runeextract.embeddings.get_onnx_embedding`."""
    from runeextract.embeddings.onnx import get_onnx_embedding as _goe
    return _goe(*args, **kwargs)


# --- Cloud Storage Connectors ---


def StorageConnector(*args, **kwargs):
    """Lazy import for :class:`runeextract.storage.StorageConnector`."""
    from runeextract.storage.connectors import StorageConnector as _SC
    return _SC(*args, **kwargs)


def S3Connector(*args, **kwargs):
    """Lazy import for :class:`runeextract.storage.S3Connector`."""
    from runeextract.storage.connectors import S3Connector as _S3
    return _S3(*args, **kwargs)


def GCSConnector(*args, **kwargs):
    """Lazy import for :class:`runeextract.storage.GCSConnector`."""
    from runeextract.storage.connectors import GCSConnector as _GCS
    return _GCS(*args, **kwargs)


def AzureConnector(*args, **kwargs):
    """Lazy import for :class:`runeextract.storage.AzureConnector`."""
    from runeextract.storage.connectors import AzureConnector as _A
    return _A(*args, **kwargs)


def get_storage_connector(*args, **kwargs):
    """Lazy import for :func:`runeextract.storage.get_storage_connector`."""
    from runeextract.storage.connectors import get_storage_connector as _gsc
    return _gsc(*args, **kwargs)


# --- Deduplication ---


def MinHashDeduplicator(*args, **kwargs):
    """Lazy import for :class:`runeextract.dedup.MinHashDeduplicator`."""
    from runeextract.dedup.minhash import MinHashDeduplicator as _MD
    return _MD(*args, **kwargs)


def LSHDeduplicator(*args, **kwargs):
    """Lazy import for :class:`runeextract.dedup.LSHDeduplicator`."""
    from runeextract.dedup.minhash import LSHDeduplicator as _LD
    return _LD(*args, **kwargs)


def EmbeddingDeduplicator(*args, **kwargs):
    """Lazy import for :class:`runeextract.dedup.EmbeddingDeduplicator`."""
    from runeextract.dedup.minhash import EmbeddingDeduplicator as _ED
    return _ED(*args, **kwargs)


def deduplicate(*args, **kwargs):
    """Lazy import for :func:`runeextract.dedup.deduplicate`."""
    from runeextract.dedup.minhash import deduplicate as _dd
    return _dd(*args, **kwargs)


def deduplicate_documents(*args, **kwargs):
    """Lazy import for :func:`runeextract.dedup.deduplicate_documents`."""
    from runeextract.dedup.minhash import deduplicate_documents as _dd
    return _dd(*args, **kwargs)


# --- WebSocket Server ---


def ExtractionServer(*args, **kwargs):
    """Lazy import for :class:`runeextract.server.ExtractionServer`."""
    from runeextract.server import ExtractionServer as _ES
    return _ES(*args, **kwargs)


# --- Visual Document Understanding ---


def VisionAnalyzer(*args, **kwargs):
    """Lazy import for :class:`runeextract.vision.VisionAnalyzer`."""
    from runeextract.vision.analyzer import VisionAnalyzer as _VA
    return _VA(*args, **kwargs)


def ChartInterpretation(*args, **kwargs):
    """Lazy import for :class:`runeextract.vision.ChartInterpretation`."""
    from runeextract.vision.analyzer import ChartInterpretation as _CI
    return _CI(*args, **kwargs)


def FigureCaption(*args, **kwargs):
    """Lazy import for :class:`runeextract.vision.FigureCaption`."""
    from runeextract.vision.analyzer import FigureCaption as _FC
    return _FC(*args, **kwargs)


def describe_image(*args, **kwargs):
    """Lazy import for :func:`runeextract.vision.describe_image`."""
    from runeextract.vision.analyzer import describe_image as _di
    return _di(*args, **kwargs)


def interpret_chart(*args, **kwargs):
    """Lazy import for :func:`runeextract.vision.interpret_chart`."""
    from runeextract.vision.analyzer import interpret_chart as _ic
    return _ic(*args, **kwargs)


def caption_figure(*args, **kwargs):
    """Lazy import for :func:`runeextract.vision.caption_figure`."""
    from runeextract.vision.analyzer import caption_figure as _cf
    return _cf(*args, **kwargs)


# --- Document Graph / GraphRAG ---


def GraphNode(*args, **kwargs):
    """Lazy import for :class:`runeextract.graph.GraphNode`."""
    from runeextract.graph.builder import GraphNode as _GN
    return _GN(*args, **kwargs)


def GraphEdge(*args, **kwargs):
    """Lazy import for :class:`runeextract.graph.GraphEdge`."""
    from runeextract.graph.builder import GraphEdge as _GE
    return _GE(*args, **kwargs)


def DocumentGraph(*args, **kwargs):
    """Lazy import for :class:`runeextract.graph.DocumentGraph`."""
    from runeextract.graph.builder import DocumentGraph as _DG
    return _DG(*args, **kwargs)


def GraphBuilder(*args, **kwargs):
    """Lazy import for :class:`runeextract.graph.GraphBuilder`."""
    from runeextract.graph.builder import GraphBuilder as _GB
    return _GB(*args, **kwargs)


def build_document_graph(*args, **kwargs):
    """Lazy import for :func:`runeextract.graph.build_document_graph`."""
    from runeextract.graph.builder import build_document_graph as _bdg
    return _bdg(*args, **kwargs)


def query_graph(*args, **kwargs):
    """Lazy import for :func:`runeextract.graph.query_graph`."""
    from runeextract.graph.builder import query_graph as _qg
    return _qg(*args, **kwargs)


# --- Pre-signed URL Extraction ---


def extract_from_presigned_url(*args, **kwargs):
    """Lazy import for :func:`runeextract.storage.presigned.extract_from_presigned_url`."""
    from runeextract.storage.presigned import extract_from_presigned_url as _epu
    return _epu(*args, **kwargs)


# --- Table of Contents ---


def TOCEntry(*args, **kwargs):
    """Lazy import for :class:`runeextract.toc.TOCEntry`."""
    from runeextract.toc import TOCEntry as _TE
    return _TE(*args, **kwargs)


def TOCParser(*args, **kwargs):
    """Lazy import for :class:`runeextract.toc.TOCParser`."""
    from runeextract.toc import TOCParser as _TP
    return _TP(*args, **kwargs)


def extract_toc(*args, **kwargs):
    """Lazy import for :func:`runeextract.toc.extract_toc`."""
    from runeextract.toc import extract_toc as _et
    return _et(*args, **kwargs)


def toc_to_markdown(*args, **kwargs):
    """Lazy import for :func:`runeextract.toc.toc_to_markdown`."""
    from runeextract.toc import toc_to_markdown as _ttm
    return _ttm(*args, **kwargs)


def toc_to_dict(*args, **kwargs):
    """Lazy import for :func:`runeextract.toc.toc_to_dict`."""
    from runeextract.toc import toc_to_dict as _ttd
    return _ttd(*args, **kwargs)


def toc_to_json(*args, **kwargs):
    """Lazy import for :func:`runeextract.toc.toc_to_json`."""
    from runeextract.toc import toc_to_json as _ttj
    return _ttj(*args, **kwargs)


# --- Multi-Language OCR ---


def OCRLanguageDetector(*args, **kwargs):
    """Lazy import for :class:`runeextract.ocr.OCRLanguageDetector`."""
    from runeextract.ocr import OCRLanguageDetector as _OLD
    return _OLD(*args, **kwargs)


def detect_ocr_language(*args, **kwargs):
    """Lazy import for :func:`runeextract.ocr.detect_ocr_language`."""
    from runeextract.ocr import detect_ocr_language as _dol
    return _dol(*args, **kwargs)


def get_tesseract_langs(*args, **kwargs):
    """Lazy import for :func:`runeextract.ocr.get_tesseract_langs`."""
    from runeextract.ocr import get_tesseract_langs as _gtl
    return _gtl(*args, **kwargs)


def get_ocr_languages(*args, **kwargs):
    """Lazy import for :func:`runeextract.ocr.get_ocr_languages`."""
    from runeextract.ocr import get_ocr_languages as _gol
    return _gol(*args, **kwargs)


# --- Fast Mode / Quality Levels ---


def _lazy_import_quality():
    """Lazy import for quality module."""
    from runeextract.quality import FastMode as _FM
    from runeextract.quality import QualityLevel as _QL
    from runeextract.quality import configure_quality as _cq
    return _FM, _QL, _cq


class _QualityLevel:
    """Proxy that supports both QualityLevel.HIGH and QualityLevel('high')."""
    _cls = None

    @classmethod
    def _get_cls(cls):
        if cls._cls is None:
            cls._cls = _lazy_import_quality()[1]
        return cls._cls

    def __getattr__(self, name):
        return getattr(self._get_cls(), name)

    def __call__(self, *args, **kwargs):
        return self._get_cls()(*args, **kwargs)

    def __repr__(self):
        return repr(self._get_cls())


QualityLevel = _QualityLevel()


def FastMode(*args, **kwargs):
    """Lazy import for :class:`runeextract.quality.FastMode`."""
    return _lazy_import_quality()[0](*args, **kwargs)


def configure_quality(*args, **kwargs):
    """Lazy import for :func:`runeextract.quality.configure_quality`."""
    return _lazy_import_quality()[2](*args, **kwargs)
