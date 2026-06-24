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
    # 🚀 Feature 3: Enhanced provenance
    bounding_box: Optional[Dict[str, float]] = None
    extracted_at: Optional[str] = None
    retrieval_rank: Optional[int] = None
    similarity_score: Optional[float] = None


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
    # 🚀 Feature 5: Cost tracking
    cost: float = 0.0
    total_session_cost: float = 0.0
    # 🚀 Feature 4: Multi-modal images
    images: List[Dict[str, str]] = field(default_factory=list)

    @property
    def sources(self) -> List[Dict[str, Any]]:
        """Shorthand: return list of simplified source info from citations."""
        return [
            {"source": c.source, "page": c.page, "text": c.text, "score": c.relevance_score}
            for c in self.citations
        ]
