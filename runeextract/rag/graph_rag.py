"""GraphRAG — entity extraction + graph traversal for enhanced RAG queries."""

import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class GraphRAGQuery:
    """Enhances RAG queries with knowledge graph context.

    Usage:
        graph_rag = GraphRAGQuery(rag, graph_builder)
        result = graph_rag.query("Who founded Company X?")
    """

    def __init__(self, rag: Any, graph_builder: Optional[Any] = None):
        self._rag = rag
        self._graph_builder = graph_builder
        self._graph = None

    def build_graph(self, docs: List[Any]):
        if not self._graph_builder:
            logger.warning("No GraphBuilder provided — skipping graph construction")
            return
        combined_text = "\n\n".join(d.text for d in docs if hasattr(d, "text"))
        self._graph = self._graph_builder.build(combined_text)

    def _extract_entities_from_question(self, question: str) -> List[str]:
        import re
        entities = []
        if self._graph:
            for node in self._graph.nodes:
                if node.label.lower() in question.lower():
                    entities.append(node.label)
        return entities

    def _get_related_chunks(self, question: str) -> List[str]:
        entities = self._extract_entities_from_question(question)
        if not entities or not self._graph:
            return []
        related_nodes: Set[str] = set()
        for entity in entities:
            for node in self._graph.nodes:
                if entity.lower() in node.label.lower():
                    related_nodes.add(node.label)
                    for edge in self._graph.edges:
                        if edge.source == node.id or edge.target == node.id:
                            for n in self._graph.nodes:
                                if n.id in (edge.source, edge.target) and n.id != node.id:
                                    related_nodes.add(n.label)
        return list(related_nodes)

    def query(self, question: str, **kwargs) -> Any:
        related = self._get_related_chunks(question)
        if related:
            context_hint = "Related entities: " + ", ".join(related[:10])
            enhanced = f"{question}\n\n{context_hint}"
            return self._rag.query(enhanced, **kwargs)
        return self._rag.query(question, **kwargs)

    def query_with_graph_traversal(self, question: str, **kwargs) -> Any:
        return self.query(question, **kwargs)
