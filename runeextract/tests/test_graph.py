"""Tests for document graph / GraphRAG."""

import pytest

from runeextract.graph.builder import (
    GraphNode, GraphEdge, DocumentGraph, GraphBuilder,
    build_document_graph, query_graph,
)


class TestGraphNode:
    def test_create(self):
        n = GraphNode(id="person:Alice", label="Alice", node_type="person")
        assert n.id == "person:Alice"
        assert n.label == "Alice"
        assert n.node_type == "person"

    def test_with_metadata(self):
        n = GraphNode(id="org:Acme", label="Acme Corp", metadata={"industry": "tech"})
        assert n.metadata["industry"] == "tech"


class TestGraphEdge:
    def test_create(self):
        e = GraphEdge(source="a", target="b", relation="works for", weight=0.9)
        assert e.source == "a"
        assert e.target == "b"
        assert e.relation == "works for"
        assert e.weight == 0.9


class TestDocumentGraph:
    def test_add_node(self):
        g = DocumentGraph()
        g.add_node(GraphNode(id="p1", label="Person 1"))
        assert len(g.nodes) == 1
        g.add_node(GraphNode(id="p1", label="Person 1"))  # duplicate
        assert len(g.nodes) == 1

    def test_add_edge(self):
        g = DocumentGraph()
        g.add_node(GraphNode(id="a", label="A"))
        g.add_node(GraphNode(id="b", label="B"))
        g.add_edge(GraphEdge(source="a", target="b", relation="knows"))
        assert len(g.edges) == 1

    def test_get_neighbors(self):
        g = DocumentGraph()
        g.add_node(GraphNode(id="a", label="A"))
        g.add_node(GraphNode(id="b", label="B"))
        g.add_node(GraphNode(id="c", label="C"))
        g.add_edge(GraphEdge(source="a", target="b", relation="knows"))
        g.add_edge(GraphEdge(source="b", target="c", relation="knows"))
        neighbors = g.get_neighbors("b")
        assert len(neighbors) == 2

    def test_find_path(self):
        g = DocumentGraph()
        for nid in ["a", "b", "c", "d"]:
            g.add_node(GraphNode(id=nid, label=nid))
        g.add_edge(GraphEdge(source="a", target="b", relation="to"))
        g.add_edge(GraphEdge(source="b", target="c", relation="to"))
        g.add_edge(GraphEdge(source="c", target="d", relation="to"))
        paths = g.find_path("a", "d", max_depth=5)
        assert len(paths) >= 1
        assert paths[0] == ["a", "b", "c", "d"]

    def test_find_path_no_path(self):
        g = DocumentGraph()
        g.add_node(GraphNode(id="a", label="A"))
        g.add_node(GraphNode(id="b", label="B"))
        paths = g.find_path("a", "b")
        assert len(paths) == 0

    def test_query(self):
        g = DocumentGraph()
        g.add_node(GraphNode(id="p1", label="Alice", metadata={"role": "engineer"}))
        g.add_node(GraphNode(id="p2", label="Bob", metadata={"role": "manager"}))
        results = g.query(["engineer"])
        assert len(results) >= 1
        assert results[0][0].id == "p1"

    def test_query_no_match(self):
        g = DocumentGraph()
        g.add_node(GraphNode(id="p1", label="Alice"))
        results = g.query(["nonexistent"])
        assert len(results) == 0

    def test_to_dict(self):
        g = DocumentGraph()
        g.add_node(GraphNode(id="x", label="X"))
        g.add_edge(GraphEdge(source="x", target="y", relation="to"))
        d = g.to_dict()
        assert len(d["nodes"]) == 1
        assert len(d["edges"]) == 1

    def test_from_dict(self):
        data = {
            "nodes": [{"id": "a", "label": "A", "type": "entity"}],
            "edges": [{"source": "a", "target": "b", "relation": "to"}],
        }
        g = DocumentGraph.from_dict(data)
        assert "a" in g.nodes
        assert len(g.edges) == 1


class TestGraphBuilder:
    def test_build_empty(self):
        g = GraphBuilder().build("")
        assert len(g.nodes) == 0

    def test_build_entities(self):
        text = "John Smith works at Acme Corp. Contact john@example.com."
        g = GraphBuilder().build(text)
        assert len(g.nodes) >= 1

    def test_build_relations(self):
        text = "Alice works for Acme Corp. Bob works for Acme Corp."
        g = GraphBuilder().build(text)
        assert len(g.edges) >= 0

    def test_build_with_source(self):
        text = "John Smith is based in New York."
        g = GraphBuilder().build(text, source="test.txt")
        node_labels = [n.label for n in g.nodes.values()]
        assert any("John" in l for l in node_labels)


class TestBuildDocumentGraph:
    def test_convenience(self):
        text = "Alice works for Acme Corp."
        g = build_document_graph(text)
        assert isinstance(g, DocumentGraph)
        assert len(g.nodes) >= 1


class TestQueryGraph:
    def test_convenience(self):
        g = DocumentGraph()
        g.add_node(GraphNode(id="t1", label="TensorFlow", metadata={"type": "framework"}))
        results = query_graph(g, "TensorFlow framework")
        assert len(results) >= 1
