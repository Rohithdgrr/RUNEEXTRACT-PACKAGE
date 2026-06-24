"""
Domain-specific RAG configuration presets.

Provides ready-made ``DomainConfig`` presets for financial, legal, medical,
and academic documents, plus a registry that allows custom overrides.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class DomainConfig:
    """Configuration preset for a specific document domain.

    Attributes:
        chunking: Chunking strategy (``"auto"``, ``"by_heading"``, etc.).
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between chunks.
        reranker: Reranker model spec or ``None``.
        embedding: Embedding model spec or ``"auto"``.
        system_prompt: Optional custom system prompt for answer generation.
        extra: Any other keyword arguments to pass to ``AutoRAG``.
    """
    chunking: str = "auto"
    chunk_size: int = 1000
    chunk_overlap: int = 100
    reranker: Optional[str] = None
    embedding: str = "auto"
    system_prompt: str = ""
    extra: Dict = field(default_factory=dict)


_DEFAULT_TEMPLATES: Dict[str, DomainConfig] = {
    "financial": DomainConfig(
        chunking="by_heading",
        chunk_size=800,
        chunk_overlap=50,
        reranker="cross-encoder/ms-marco-MiniLM-L-6-v2",
        embedding="balanced",
        system_prompt=(
            "You are a financial analyst. Answer using only the provided "
            "context. Cite specific numbers, dates, and sources. "
            "If the context lacks financial data, state that clearly."
        ),
    ),
    "legal": DomainConfig(
        chunking="by_heading",
        chunk_size=1200,
        chunk_overlap=100,
        reranker="cross-encoder/ms-marco-MiniLM-L-6-v2",
        embedding="balanced",
        system_prompt=(
            "You are a legal assistant. Answer using only the provided "
            "context. Reference specific clauses, statutes, or case law "
            "by citation number. Avoid speculation."
        ),
    ),
    "medical": DomainConfig(
        chunking="sentence_window",
        chunk_size=600,
        chunk_overlap=30,
        reranker="sentence-transformers/all-MiniLM-L6-v2",
        embedding="accurate",
        system_prompt=(
            "You are a medical information specialist. Answer using only "
            "the provided context. Disclaim if the context is insufficient "
            "for a clinical decision. Do not provide diagnosis."
        ),
    ),
    "academic": DomainConfig(
        chunking="hierarchical",
        chunk_size=1000,
        chunk_overlap=50,
        reranker="cross-encoder/ms-marco-MiniLM-L-6-v2",
        embedding="balanced",
        system_prompt=(
            "You are a research assistant. Answer using only the provided "
            "context. Cite sources with [1], [2] markers. "
            "Mention methodologies, sample sizes, and statistical results."
        ),
    ),
}


class DomainTemplates:
    """Registry of domain-specific RAG configuration presets."""

    _templates: Dict[str, DomainConfig] = dict(_DEFAULT_TEMPLATES)

    @classmethod
    def get(cls, name: str) -> DomainConfig:
        """Return the ``DomainConfig`` for *name*, or ``DomainConfig()`` if unknown."""
        return cls._templates.get(name, DomainConfig())

    @classmethod
    def list(cls) -> Dict[str, DomainConfig]:
        """Return a copy of all registered templates."""
        return dict(cls._templates)

    @classmethod
    def register(cls, name: str, config: DomainConfig) -> None:
        """Register (or override) a domain template."""
        cls._templates[name] = config

    @classmethod
    def reset(cls) -> None:
        """Restore all templates to their defaults."""
        cls._templates = dict(_DEFAULT_TEMPLATES)
