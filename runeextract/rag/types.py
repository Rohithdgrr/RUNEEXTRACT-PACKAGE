"""
Shared dataclasses for the RAG pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Citation:
    text: str
    source: str
    page: Optional[int] = None
    chunk_index: int = 0
    relevance_score: float = 0.0


@dataclass
class ChunkWithScore:
    text: str
    score: float
    source: str = ""
    source_type: str = ""
    document_id: str = ""
    chunk_id: str = ""
    page: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResult:
    answer: str
    citations: List[Citation] = field(default_factory=list)
    confidence: float = 0.0
    retrieved_chunks: List[ChunkWithScore] = field(default_factory=list)
    query_variants: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    tokens_used: Dict[str, int] = field(default_factory=dict)
