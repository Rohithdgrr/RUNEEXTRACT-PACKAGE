"""
Hierarchical / RAPTOR-style chunking.

Builds a tree of summaries from leaf-level chunks, enabling multi-hop
reasoning by retrieving context at multiple levels of abstraction.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Tuple, Union

from runeextract.utils.maturity import experimental

logger = logging.getLogger(__name__)


@dataclass
class SummaryNode:
    """A node in the hierarchical summary tree."""
    text: str
    level: int
    node_id: str
    children: List["SummaryNode"] = field(default_factory=list)
    chunk_indices: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    summary_score: float = 0.0


@dataclass
class HierarchicalResult:
    """Result from a hierarchical retrieval."""
    nodes: List[SummaryNode]
    levels: List[int]
    texts: List[str]
    scores: List[float]
    tree: Optional[SummaryNode] = None
    num_nodes: int = 0


@experimental(name="rag.hierarchical")
class HierarchicalChunker:
    """Build and query a hierarchical summary tree (RAPTOR-style).

    The tree is built by:
      1. Clustering leaf chunks (by embedding similarity or sequential grouping)
      2. Summarizing each cluster to create level-1 parent nodes
      3. Recursing until a single root summary remains

    Args:
        summarizer: Optional callable ``fn(texts: List[str]) -> str`` that
            produces a summary from a list of texts. If None, concatenation
            is used.
        embedder: Optional callable ``fn(texts: List[str]) -> List[List[float]]``
            that produces embeddings. If None, sequential clustering is used.
        cluster_size: Target number of chunks per cluster (default 5).
        max_levels: Maximum tree depth (default 3).
        similarity_threshold: Cosine similarity threshold for clustering
            when embedder is provided (default 0.5).
    """

    def __init__(
        self,
        summarizer: Optional[Callable[[List[str]], str]] = None,
        embedder: Optional[Callable[[List[str]], List[List[float]]]] = None,
        cluster_size: int = 5,
        max_levels: int = 3,
        similarity_threshold: float = 0.5,
    ):
        self.summarizer = summarizer
        self.embedder = embedder
        self.cluster_size = cluster_size
        self.max_levels = max_levels
        self.similarity_threshold = similarity_threshold
        self._tree: Optional[SummaryNode] = None
        self._all_nodes: List[SummaryNode] = []
        self._leaf_count = 0

    @property
    def tree(self) -> Optional[SummaryNode]:
        """Root of the built summary tree (None until build_tree is called)."""
        return self._tree

    def build_tree(self, texts: List[str], metadata: Optional[List[Dict]] = None) -> SummaryNode:
        """Build a hierarchical summary tree from leaf texts.

        Args:
            texts: List of leaf-level text chunks.
            metadata: Optional list of metadata dicts for each chunk.

        Returns:
            Root SummaryNode of the tree.
        """
        self._all_nodes = []
        self._leaf_count = len(texts)

        if not texts:
            raise ValueError("Cannot build tree from empty text list")

        # Create leaf nodes
        leaves = [
            SummaryNode(
                text=t,
                level=0,
                node_id=f"L{i}",
                chunk_indices=[i],
                metadata=(metadata[i] if metadata and i < len(metadata) else {}),
            )
            for i, t in enumerate(texts)
        ]
        self._all_nodes.extend(leaves)

        # Recursively build upper levels
        current_level_nodes = leaves
        level = 0

        while len(current_level_nodes) > 1 and level < self.max_levels:
            level += 1
            clusters = self._cluster(current_level_nodes, level)
            parent_nodes = []

            for cluster_idx, cluster in enumerate(clusters):
                cluster_texts = [n.text for n in cluster]
                summary = self._summarize(cluster_texts)
                parent = SummaryNode(
                    text=summary,
                    level=level,
                    node_id=f"L{level}_{cluster_idx}",
                    children=cluster,
                    chunk_indices=[ci for n in cluster for ci in n.chunk_indices],
                    metadata={"num_children": len(cluster), "level": level},
                )
                parent_nodes.append(parent)
                self._all_nodes.append(parent)

            current_level_nodes = parent_nodes

        # Root is the single top-level node (or merged if multiple)
        if len(current_level_nodes) == 1:
            self._tree = current_level_nodes[0]
            # If the single node is at level 0 (only one leaf), wrap it
            if self._tree.level == 0 and len(texts) > 1:
                summary = self._summarize([self._tree.text])
                self._tree = SummaryNode(
                    text=summary,
                    level=1,
                    node_id="ROOT",
                    children=[current_level_nodes[0]],
                    chunk_indices=list(range(len(texts))),
                    metadata={"num_children": 1},
                )
                self._all_nodes.append(self._tree)
        elif len(current_level_nodes) == 0:
            self._tree = SummaryNode(
                text=self._summarize(texts),
                level=1,
                node_id="ROOT",
                children=[],
                chunk_indices=list(range(len(texts))),
            )
            self._all_nodes.append(self._tree)
        else:
            # Collapse remaining into a single root
            all_texts = [n.text for n in current_level_nodes]
            root_summary = self._summarize(all_texts)
            self._tree = SummaryNode(
                text=root_summary,
                level=level + 1,
                node_id="ROOT",
                children=current_level_nodes,
                chunk_indices=[ci for n in current_level_nodes for ci in n.chunk_indices],
                metadata={"num_children": len(current_level_nodes)},
            )
            self._all_nodes.append(self._tree)

        return self._tree

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        level_weights: Optional[List[float]] = None,
    ) -> HierarchicalResult:
        """Retrieve the most relevant nodes across all tree levels.

        Args:
            query: The query string.
            top_k: Number of nodes to return across all levels.
            level_weights: Weight per level for scoring (default: higher
                levels get lower weight to prefer granular chunks).

        Returns:
            HierarchicalResult with scored nodes from all levels.
        """
        if self._tree is None:
            raise RuntimeError("No tree built yet; call build_tree() first")

        if level_weights is None:
            level_weights = [1.0, 0.8, 0.6, 0.5]

        query_lower = query.lower()
        query_tokens = set(query_lower.split())

        scored: List[Tuple[SummaryNode, float]] = []
        for node in self._all_nodes:
            weight = level_weights[node.level] if node.level < len(level_weights) else 0.3
            score = self._score_node(node, query_lower, query_tokens) * weight
            if score > 0:
                scored.append((node, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Deduplicate by chunk_indices — prefer higher-scored
        seen_chunks: set = set()
        unique: List[Tuple[SummaryNode, float]] = []
        for node, score in scored:
            key = tuple(sorted(node.chunk_indices))
            if key not in seen_chunks:
                seen_chunks.add(key)
                unique.append((node, score))
                if len(unique) >= top_k:
                    break

        nodes = [n for n, _ in unique]
        scores = [s for _, s in unique]

        return HierarchicalResult(
            nodes=nodes,
            levels=[n.level for n in nodes],
            texts=[n.text for n in nodes],
            scores=scores,
            tree=self._tree,
            num_nodes=len(self._all_nodes),
        )

    def _cluster(
        self,
        nodes: List[SummaryNode],
        level: int,
    ) -> List[List[SummaryNode]]:
        """Cluster nodes at a given level.

        Uses embeddings if available, otherwise sequential grouping.
        """
        if len(nodes) <= self.cluster_size:
            return [nodes]

        if self.embedder is not None:
            return self._cluster_by_similarity(nodes)

        return self._cluster_sequential(nodes)

    def _cluster_sequential(self, nodes: List[SummaryNode]) -> List[List[SummaryNode]]:
        """Group adjacent nodes into fixed-size clusters."""
        clusters = []
        for i in range(0, len(nodes), self.cluster_size):
            cluster = nodes[i:i + self.cluster_size]
            if cluster:
                clusters.append(cluster)
        return clusters

    def _cluster_by_similarity(self, nodes: List[SummaryNode]) -> List[List[SummaryNode]]:
        """Cluster nodes by embedding similarity using greedy agglomerative."""
        try:
            import numpy as np
        except ImportError:
            logger.debug("numpy not available; falling back to sequential clustering")
            return self._cluster_sequential(nodes)

        texts = [n.text for n in nodes]
        embeddings = self.embedder(texts)
        emb_matrix = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        emb_matrix = emb_matrix / norms

        # Greedy clustering: start with first as cluster seed,
        # add similar items until cluster_size or threshold
        assigned = [False] * len(nodes)
        clusters = []

        for i in range(len(nodes)):
            if assigned[i]:
                continue
            cluster = [nodes[i]]
            assigned[i] = True

            for j in range(i + 1, len(nodes)):
                if assigned[j]:
                    continue
                if len(cluster) >= self.cluster_size:
                    break
                sim = float(np.dot(emb_matrix[i], emb_matrix[j]))
                if sim >= self.similarity_threshold:
                    cluster.append(nodes[j])
                    assigned[j] = True

            clusters.append(cluster)

        return clusters

    def _score_node(
        self,
        node: SummaryNode,
        query_lower: str,
        query_tokens: set,
    ) -> float:
        """Score a node's relevance to a query using text overlap."""
        text_lower = node.text.lower()

        # Token overlap score
        text_tokens = set(text_lower.split())
        if not query_tokens:
            return 0.0
        overlap = len(query_tokens & text_tokens)
        token_score = overlap / len(query_tokens)

        # Exact phrase match bonus
        phrase_bonus = 1.5 if query_lower in text_lower else 0.0

        # Length normalization (prefer concise matches)
        length_penalty = min(1.0, 500.0 / max(len(node.text), 1))

        return (token_score + phrase_bonus) * length_penalty

    def _summarize(self, texts: List[str]) -> str:
        """Summarize a list of texts into a single condensed version.

        Uses the configured summarizer callable, or falls back to
        concatenation with deduplication.
        """
        if self.summarizer is not None:
            try:
                return self.summarizer(texts)
            except Exception as exc:
                logger.debug(f"Summarizer failed: {exc}; using concatenation fallback")

        # Fallback: concatenate with deduplication
        seen = set()
        parts = []
        for t in texts:
            key = t.strip()[:100]
            if key not in seen:
                seen.add(key)
                parts.append(t.strip())

        joined = "\n\n".join(parts)
        if len(joined) > 2000:
            joined = joined[:2000] + "..."

        return joined

    def print_tree(self, node: Optional[SummaryNode] = None, indent: int = 0):
        """Print a human-readable view of the summary tree."""
        if node is None:
            node = self._tree
            if node is None:
                print("(no tree)")
                return

        prefix = "  " * indent
        text_preview = node.text[:80].replace("\n", " ")
        children_info = f" [{len(node.children)} children]" if node.children else ""
        print(f"{prefix}[L{node.level}] {text_preview}{children_info}")

        for child in node.children:
            self.print_tree(child, indent + 1)
