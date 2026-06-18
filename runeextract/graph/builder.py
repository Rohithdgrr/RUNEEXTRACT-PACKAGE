"""Document graph builder — extract entities and relationships to build a knowledge graph.

Entity extraction uses either:
- A simple regex/heuristic approach (no deps)
- An AI provider for NLU-quality extraction (optional)
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    id: str
    label: str
    node_type: str = "entity"
    metadata: dict = field(default_factory=dict)
    embeddings: Optional[List[float]] = None


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)


@dataclass
class DocumentGraph:
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: List[GraphEdge] = field(default_factory=list)

    def add_node(self, node: GraphNode):
        if node.id not in self.nodes:
            self.nodes[node.id] = node

    def add_edge(self, edge: GraphEdge):
        self.edges.append(edge)

    def get_neighbors(self, node_id: str) -> List[Tuple[str, str, float]]:
        results = []
        for e in self.edges:
            if e.source == node_id:
                results.append((e.target, e.relation, e.weight))
            if e.target == node_id:
                results.append((e.source, e.relation, e.weight))
        return results

    def find_path(self, source: str, target: str, max_depth: int = 3) -> List[List[str]]:
        paths = []
        visited = set()

        def _dfs(current: str, path: List[str]):
            if len(path) > max_depth:
                return
            if current == target:
                paths.append(list(path))
                return
            visited.add(current)
            for neighbor, _, _ in self.get_neighbors(current):
                if neighbor not in visited:
                    path.append(neighbor)
                    _dfs(neighbor, path)
                    path.pop()
            visited.discard(current)

        _dfs(source, [source])
        return paths

    def query(self, query_terms: List[str]) -> List[Tuple[GraphNode, float]]:
        scored = []
        for node_id, node in self.nodes.items():
            score = 0.0
            ql = query_terms
            nl = node.label.lower()
            score += sum(2.0 for qt in ql if qt.lower() in nl)
            for val in node.metadata.values():
                if isinstance(val, str):
                    score += sum(1.0 for qt in ql if qt.lower() in val.lower())
            if score > 0:
                scored.append((node, score))
        return sorted(scored, key=lambda x: -x[1])

    def to_dict(self) -> dict:
        return {
            "nodes": [
                {"id": n.id, "label": n.label, "type": n.node_type, "metadata": n.metadata}
                for n in self.nodes.values()
            ],
            "edges": [
                {"source": e.source, "target": e.target, "relation": e.relation, "weight": e.weight}
                for e in self.edges
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentGraph":
        g = cls()
        for nd in data.get("nodes", []):
            g.add_node(GraphNode(id=nd["id"], label=nd["label"], node_type=nd.get("type", "entity"), metadata=nd.get("metadata", {})))
        for ed in data.get("edges", []):
            g.add_edge(GraphEdge(source=ed["source"], target=ed["target"], relation=ed["relation"], weight=ed.get("weight", 1.0)))
        return g


class GraphBuilder:
    """Build a knowledge graph from document text.

    Uses heuristic entity extraction by default. Optionally uses
    AIProcessor for higher-quality extraction.
    """

    # Common entity patterns
    ENTITY_PATTERNS = [
        (r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", "person"),
        (r"\b(?:https?://|www\.)\S+\b", "url"),
        (r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "ip_address"),
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "email"),
        (r"\b\d{3}-\d{2}-\d{4}\b", "ssn"),
        (r"\b\d{10}\b", "phone"),
    ]

    # Common relation keywords
    RELATION_KEYWORDS = {
        "works for": ["works for", "employed by", "works at"],
        "located in": ["located in", "based in", "situated in"],
        "part of": ["part of", "belongs to", "member of"],
        "created by": ["created by", "authored by", "written by"],
        "acquired": ["acquired", "purchased", "bought"],
        "led by": ["led by", "headed by", "chaired by"],
        "invested in": ["invested in", "funded"],
    }

    def __init__(self, use_ai: bool = False, provider: str = "openai", model: str = "gpt-4o-mini"):
        self.use_ai = use_ai
        self.provider = provider
        self.model = model
        self._graph = DocumentGraph()

    def build(self, text: str, source: str = "") -> DocumentGraph:
        self._graph = DocumentGraph()
        self._extract_entities(text)
        self._extract_relations(text)
        return self._graph

    def _extract_entities(self, text: str):
        for pattern, etype in self.ENTITY_PATTERNS:
            for m in re.finditer(pattern, text):
                label = m.group()
                node_id = f"{etype}:{label}"
                node = GraphNode(
                    id=node_id,
                    label=label,
                    node_type=etype,
                    metadata={"source": text[:50]},
                )
                self._graph.add_node(node)

        if self.use_ai:
            self._ai_extract_entities(text)

    def _ai_extract_entities(self, text: str):
        try:
            from runeextract.processors.ai import AIProcessor
            ai = AIProcessor(provider=self.provider, model=self.model)
            prompt = (
                "Extract all named entities from the following text. "
                "Return them as a JSON list of {label, type} objects. "
                "Types: person, organization, location, date, product, event, concept.\n\n"
                f"Text: {text[:2000]}"
            )
            response = ai._call(messages=[{"role": "user", "content": prompt}], model=self.model)
            raw = response.get("text", "") or response.get("content", "") or ""
            import json
            match = re.search(r'\[.*?\]', raw, re.DOTALL)
            if match:
                entities = json.loads(match.group())
                for ent in entities:
                    label = ent.get("label", "")
                    etype = ent.get("type", "entity")
                    node_id = f"ai:{etype}:{label}"
                    self._graph.add_node(GraphNode(id=node_id, label=label, node_type=etype))
        except Exception as e:
            logger.debug(f"AI entity extraction failed: {e}")

    def _extract_relations(self, text: str):
        sentences = re.split(r'[.!?]+', text)
        nodes_list = list(self._graph.nodes.values())

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            for rel_name, keywords in self.RELATION_KEYWORDS.items():
                for kw in keywords:
                    if kw in sentence.lower():
                        parts = sentence.lower().split(kw)
                        if len(parts) != 2:
                            continue
                        left, right = parts[0].strip(), parts[1].strip()
                        left_words = left.split()
                        right_words = right.split()
                        source_node = self._find_best_node(left_words)
                        target_node = self._find_best_node(right_words)
                        if source_node and target_node and source_node != target_node:
                            self._graph.add_edge(GraphEdge(
                                source=source_node.id,
                                target=target_node.id,
                                relation=rel_name,
                                weight=1.0,
                            ))

    def _find_best_node(self, words: List[str]) -> Optional[GraphNode]:
        best_score = 0
        best_node = None
        for node in self._graph.nodes.values():
            label_words = node.label.lower().split()
            score = sum(1 for w in words if w in label_words)
            if score > best_score:
                best_score = score
                best_node = node
        return best_node if best_score > 0 else None


def build_document_graph(text: str, source: str = "", use_ai: bool = False) -> DocumentGraph:
    builder = GraphBuilder(use_ai=use_ai)
    return builder.build(text, source=source)


def query_graph(graph: DocumentGraph, query: str) -> List[Tuple[GraphNode, float]]:
    return graph.query(query.split())
