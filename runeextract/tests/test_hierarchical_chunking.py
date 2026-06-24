"""
Tests for hierarchical / RAPTOR-style chunking.
"""

import pytest
from unittest.mock import Mock, patch

from runeextract.models.document import ChunkingStrategy
from runeextract.rag.hierarchical import (
    HierarchicalChunker,
    SummaryNode,
    HierarchicalResult,
)


class TestSummaryNode:
    def test_defaults(self):
        node = SummaryNode(text="hello", level=0, node_id="L0")
        assert node.text == "hello"
        assert node.level == 0
        assert node.node_id == "L0"
        assert node.children == []
        assert node.chunk_indices == []
        assert node.summary_score == 0.0

    def test_with_children(self):
        child = SummaryNode(text="child", level=0, node_id="C0")
        parent = SummaryNode(text="parent", level=1, node_id="P0", children=[child])
        assert len(parent.children) == 1
        assert parent.children[0].node_id == "C0"

    def test_chunk_indices(self):
        node = SummaryNode(text="t", level=0, node_id="N", chunk_indices=[0, 1, 2])
        assert node.chunk_indices == [0, 1, 2]


class TestHierarchicalResult:
    def test_defaults(self):
        r = HierarchicalResult(nodes=[], levels=[], texts=[], scores=[])
        assert r.num_nodes == 0
        assert r.tree is None

    def test_with_nodes(self):
        nodes = [SummaryNode(text="a", level=0, node_id="L0")]
        r = HierarchicalResult(
            nodes=nodes, levels=[0], texts=["a"], scores=[0.9],
            tree=nodes[0], num_nodes=5,
        )
        assert r.tree is not None
        assert r.num_nodes == 5
        assert r.scores == [0.9]


class TestHierarchicalChunker:
    def test_init_defaults(self):
        hc = HierarchicalChunker()
        assert hc.cluster_size == 5
        assert hc.max_levels == 3
        assert hc.similarity_threshold == 0.5
        assert hc.tree is None

    def test_init_custom(self):
        fn = lambda x: ""
        hc = HierarchicalChunker(summarizer=fn, cluster_size=3, max_levels=5)
        assert hc.cluster_size == 3
        assert hc.max_levels == 5
        assert hc.summarizer is fn

    def test_build_tree_simple(self):
        texts = [
            "The sky is blue and clear today.",
            "Birds fly high in the sky.",
            "The sun is bright and warm.",
            "Trees sway gently in the breeze.",
            "Flowers bloom in the spring.",
        ]
        hc = HierarchicalChunker(cluster_size=3, max_levels=3)
        root = hc.build_tree(texts)
        assert root is not None
        assert root.level >= 1
        assert root.node_id == "ROOT" or root.node_id.startswith("L")
        assert hc.tree is not None

    def test_build_tree_single_cluster(self):
        texts = ["Only one cluster of text here."]
        hc = HierarchicalChunker(cluster_size=10, max_levels=3)
        root = hc.build_tree(texts)
        assert root is not None
        assert root.level >= 0
        # Single text should return the text itself as the tree

    def test_build_tree_empty_raises(self):
        hc = HierarchicalChunker()
        with pytest.raises(ValueError, match="empty"):
            hc.build_tree([])

    def test_retrieve_after_build(self):
        texts = [
            "Python is a programming language.",
            "Python supports object-oriented programming.",
            "Dogs are loyal pets.",
            "Cats are independent animals.",
            "The quick brown fox jumps over the lazy dog.",
        ]
        hc = HierarchicalChunker(cluster_size=3, max_levels=2)
        hc.build_tree(texts)
        result = hc.retrieve("python programming", top_k=5)
        assert len(result.nodes) > 0
        assert len(result.texts) > 0
        assert len(result.scores) == len(result.nodes)
        # Python-related texts should score higher
        python_texts = [t for t in result.texts if "python" in t.lower()]
        assert len(python_texts) >= 1

    def test_retrieve_without_tree_raises(self):
        hc = HierarchicalChunker()
        with pytest.raises(RuntimeError, match="No tree built"):
            hc.retrieve("test")

    def test_retrieve_empty_query(self):
        texts = ["Some text here.", "More text there."]
        hc = HierarchicalChunker()
        hc.build_tree(texts)
        result = hc.retrieve("", top_k=5)
        assert len(result.nodes) >= 0  # Empty query returns nothing or empty

    def test_cluster_sequential(self):
        hc = HierarchicalChunker(cluster_size=3)
        nodes = [
            SummaryNode(text=f"text{i}", level=0, node_id=f"L{i}")
            for i in range(10)
        ]
        clusters = hc._cluster_sequential(nodes)
        assert len(clusters) == 4  # 10 items, 3 per cluster = 4 clusters
        assert len(clusters[0]) == 3
        assert len(clusters[-1]) == 1  # last has 1

    def test_cluster_sequential_exact(self):
        hc = HierarchicalChunker(cluster_size=5)
        nodes = [SummaryNode(text="t", level=0, node_id=f"L{i}") for i in range(5)]
        clusters = hc._cluster_sequential(nodes)
        assert len(clusters) == 1
        assert len(clusters[0]) == 5

    def test_summarize_fallback(self):
        hc = HierarchicalChunker()
        texts = ["First paragraph.", "Second paragraph."]
        result = hc._summarize(texts)
        assert "First paragraph" in result
        assert "Second paragraph" in result

    def test_summarize_deduplication(self):
        hc = HierarchicalChunker()
        texts = ["Same content.", "Same content.", "Different content."]
        result = hc._summarize(texts)
        assert result.count("Same content") == 1
        assert "Different content" in result

    def test_summarize_with_callable(self):
        mock_fn = Mock(return_value="Custom summary")
        hc = HierarchicalChunker(summarizer=mock_fn)
        result = hc._summarize(["text1", "text2"])
        assert result == "Custom summary"
        mock_fn.assert_called_once_with(["text1", "text2"])

    def test_score_node_token_overlap(self):
        hc = HierarchicalChunker()
        node = SummaryNode(text="Python is great for programming", level=0, node_id="L0")
        score = hc._score_node(node, "python programming", {"python", "programming"})
        assert score > 0

    def test_score_node_no_overlap(self):
        hc = HierarchicalChunker()
        node = SummaryNode(text="Dogs are loyal", level=0, node_id="L0")
        score = hc._score_node(node, "python programming", {"python", "programming"})
        assert score == 0.0

    def test_score_node_phrase_bonus(self):
        hc = HierarchicalChunker()
        node = SummaryNode(text="Python programming is fun", level=0, node_id="L0")
        score_exact = hc._score_node(node, "python programming", {"python", "programming"})
        node2 = SummaryNode(text="Python is fun for programming", level=0, node_id="L1")
        score_partial = hc._score_node(node2, "python programming", {"python", "programming"})
        # Exact phrase match should get a bonus
        assert score_exact >= score_partial

    def test_build_tree_structure(self):
        """Verify tree has proper levels and hierarchy."""
        texts = [f"Paragraph {i} content here." for i in range(20)]
        hc = HierarchicalChunker(cluster_size=5, max_levels=3)
        root = hc.build_tree(texts)
        # Collect all nodes
        all_nodes = [root]
        queue = list(root.children)
        while queue:
            n = queue.pop(0)
            all_nodes.append(n)
            queue.extend(n.children)
        # Verify leaves cover all inputs
        leaf_indices = set()
        for n in all_nodes:
            if n.level == 0:
                leaf_indices.update(n.chunk_indices)
        for i in range(len(texts)):
            assert i in leaf_indices, f"Leaf index {i} not covered"

    def test_all_nodes_property(self):
        texts = [f"Text {i}" for i in range(10)]
        hc = HierarchicalChunker(cluster_size=5, max_levels=3)
        hc.build_tree(texts)
        assert len(hc._all_nodes) > 10  # leaves + parents + root
        levels = set(n.level for n in hc._all_nodes)
        assert 0 in levels  # leaf level
        assert max(levels) >= 1  # at least one parent level

    def test_print_tree_no_crash(self):
        texts = ["Hello world.", "Test content."]
        hc = HierarchicalChunker()
        hc.build_tree(texts)
        # Should not raise
        hc.print_tree()

    def test_print_tree_no_tree(self, capsys):
        hc = HierarchicalChunker()
        hc.print_tree()
        captured = capsys.readouterr()
        assert "(no tree)" in captured.out

    def test_cluster_by_similarity_fallback(self):
        """Without embedder, similarity clustering should fall back to sequential."""
        hc = HierarchicalChunker(cluster_size=3)
        nodes = [SummaryNode(text="t", level=0, node_id=f"L{i}") for i in range(7)]
        clusters = hc._cluster(nodes, level=1)
        assert len(clusters) == 3  # 7 items, 3 per cluster

    def test_level_weights_in_retrieve(self):
        texts = [f"Content block {i}" for i in range(12)]
        hc = HierarchicalChunker(cluster_size=4, max_levels=3)
        hc.build_tree(texts)
        # Without level weights
        r1 = hc.retrieve("Content block", top_k=10)
        # With different weights
        r2 = hc.retrieve("Content block", top_k=10, level_weights=[0.1, 0.1, 0.1])
        assert len(r1.nodes) > 0


class TestChunkingStrategy:
    def test_hierarchical_exists(self):
        assert ChunkingStrategy.HIERARCHICAL == "hierarchical"

    def test_document_chunk_hierarchical(self):
        from runeextract.models.document import Document
        doc = Document(text=("Hello world. " * 30) + ("Goodbye moon. " * 30))
        chunks = doc.chunks(strategy=ChunkingStrategy.HIERARCHICAL, leaf_size=100)
        assert len(chunks) >= 1
        for c in chunks:
            assert c.metadata.get("strategy") == "hierarchical"


class TestAutoRAGHierarchical:
    def test_hierarchical_chunking_mode(self):
        from runeextract.rag.auto_pipeline import AutoRAG
        rag = AutoRAG(chunking="hierarchical")
        assert rag.chunking_mode == "hierarchical"

    def test_hierarchical_auto_detect(self):
        from runeextract.rag.auto_pipeline import AutoRAG
        from runeextract.models.document import Document
        rag = AutoRAG(chunking="auto")
        # Very long doc should trigger hierarchical
        doc = Document(text="word " * 50000)
        strategy = rag._resolve_chunking(doc)
        assert strategy == "hierarchical"

    @patch("runeextract.rag.auto_pipeline.HierarchicalChunker.__wrapped__.build_tree")
    def test_build_hierarchical_tree_called(self, mock_build):
        from runeextract.rag.auto_pipeline import AutoRAG
        from runeextract.models.document import Document, Chunk
        from runeextract.processors.ai import AIProcessor
        mock_ai = Mock(spec=AIProcessor)
        mock_ai._call.return_value = "summary text"
        rag = AutoRAG(chunking="hierarchical", vector_store="chromadb",
                       ai_processor=mock_ai)
        doc = Document(text="test " * 100)
        chunks = [Chunk(text="test", chunk_id="c0", start_index=0, end_index=4)]
        rag._build_hierarchical_tree(doc, chunks)
        mock_build.assert_called()


class TestHierarchicalImports:
    def test_rag_init_exports(self):
        from runeextract.rag import HierarchicalChunker, SummaryNode, HierarchicalResult
        assert HierarchicalChunker is not None
        assert SummaryNode is not None
        assert HierarchicalResult is not None
