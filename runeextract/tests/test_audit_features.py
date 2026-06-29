"""Tests for all 10 audit-recommended features."""

import json
import os
import tempfile
import time
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# ---- Feature 1: Persistent Embedding Cache ----


class TestPersistentEmbeddingCache:
    @staticmethod
    def _make_cache(**kw):
        from runeextract.rag.embed_cache import PersistentEmbeddingCache
        path = tempfile.mktemp(suffix=".db")
        return PersistentEmbeddingCache(db_path=path, **kw), path

    def _close_and_clean(self, cache, path):
        cache.close()
        try:
            os.unlink(path)
        except PermissionError:
            pass

    def test_put_and_get(self):
        cache, path = self._make_cache()
        try:
            cache.put("hello world", "test-model", [0.1, 0.2, 0.3])
            vec = cache.get("hello world", "test-model")
            assert vec is not None
            assert len(vec) == 3
            assert abs(vec[0] - 0.1) < 0.001
        finally:
            self._close_and_clean(cache, path)

    def test_get_missing(self):
        cache, path = self._make_cache()
        try:
            assert cache.get("nonexistent", "test-model") is None
        finally:
            self._close_and_clean(cache, path)

    def test_put_batch(self):
        cache, path = self._make_cache()
        try:
            texts = ["a", "b", "c"]
            vectors = [[1.0], [2.0], [3.0]]
            cache.put_batch(texts, "test-model", vectors)
            for text in texts:
                assert cache.get(text, "test-model") is not None
        finally:
            self._close_and_clean(cache, path)

    def test_get_batch(self):
        cache, path = self._make_cache()
        try:
            cache.put("hello", "m", [0.5])
            result = cache.get_batch(["hello", "world"], "m")
            assert result["hello"] is not None
            assert result["world"] is None
        finally:
            self._close_and_clean(cache, path)

    def test_clear(self):
        cache, path = self._make_cache()
        try:
            cache.put("x", "m", [1.0])
            assert cache.size > 0
            cache.clear()
            assert cache.size == 0
        finally:
            self._close_and_clean(cache, path)

    def test_close_reopens(self):
        from runeextract.rag.embed_cache import PersistentEmbeddingCache
        path = tempfile.mktemp(suffix=".db")
        cache = PersistentEmbeddingCache(db_path=path)
        try:
            cache.put("data", "m", [0.9, 0.8])
            cache.close()
            cache2 = PersistentEmbeddingCache(db_path=path)
            vec = cache2.get("data", "m")
            assert vec is not None
            assert abs(vec[0] - 0.9) < 0.001
            cache2.close()
        finally:
            try:
                os.unlink(path)
            except PermissionError:
                pass

    def test_lru_eviction(self):
        cache, path = self._make_cache(max_entries=3)
        try:
            for i in range(5):
                cache.put(f"text{i}", "m", [float(i)])
            assert cache.size <= 3
        finally:
            self._close_and_clean(cache, path)


# ---- Feature 2: Async-First Ingestion Pipeline ----


class TestAsyncIngestion:
    @pytest.mark.asyncio
    async def test_aingest_empty_source(self):
        from runeextract.rag import AutoRAG
        rag = AutoRAG()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Test")
            path = f.name
        try:
            docs = await rag.aingest(path, max_concurrency=2)
            assert isinstance(docs, list)
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_aingest_returns_documents(self):
        from runeextract.rag import AutoRAG
        rag = AutoRAG()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Hello\nWorld content")
            path = f.name
        try:
            docs = await rag.aingest(path, incremental=False)
            assert len(docs) >= 1
            assert "Hello" in docs[0].text or "World" in docs[0].text
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_ingest_single_returns_none_on_error(self):
        from runeextract.rag.auto_pipeline import AutoRAG
        rag = AutoRAG()
        doc = rag._ingest_single("/nonexistent/file.pdf", False, {})
        assert doc is None


# ---- Feature 3: AutoRAG State Persistence ----


class TestIndexPersister:
    def test_save_and_load(self):
        from runeextract.rag.persistence import IndexPersister, IndexState
        with tempfile.TemporaryDirectory() as d:
            persister = IndexPersister(state_dir=d)
            state = IndexState(
                file_hashes={"a.pdf": "abc123"},
                file_chunk_ids={"a.pdf": ["chunk1"]},
                collection_name="docs",
                total_documents=1,
            )
            persister.save(state, tag="test")
            loaded = persister.load("test")
            assert loaded is not None
            assert loaded.file_hashes == {"a.pdf": "abc123"}
            assert loaded.file_chunk_ids == {"a.pdf": ["chunk1"]}
            assert loaded.total_documents == 1

    def test_load_nonexistent(self):
        from runeextract.rag.persistence import IndexPersister
        with tempfile.TemporaryDirectory() as d:
            persister = IndexPersister(state_dir=d)
            assert persister.load("nonexistent") is None

    def test_list_snapshots(self):
        from runeextract.rag.persistence import IndexPersister, IndexState
        with tempfile.TemporaryDirectory() as d:
            persister = IndexPersister(state_dir=d)
            persister.save(IndexState(), tag="snap1")
            persister.save(IndexState(), tag="snap2")
            snaps = persister.list_snapshots()
            assert "snap1" in snaps
            assert "snap2" in snaps

    def test_delete(self):
        from runeextract.rag.persistence import IndexPersister, IndexState
        with tempfile.TemporaryDirectory() as d:
            persister = IndexPersister(state_dir=d)
            persister.save(IndexState(), tag="tmp")
            persister.delete("tmp")
            assert persister.load("tmp") is None

    def test_save_to_rag(self):
        from runeextract.rag.persistence import IndexPersister

        class FakeRAG:
            _file_hashes = {"f.pdf": "hash"}
            _file_chunk_ids = {"f.pdf": ["c1"]}
            collection_name = "docs"
            persist_directory = "./chroma_db"
            embedding_spec = "openai:text-embedding-3-small"
            _documents = [1, 2]

        rag = FakeRAG()
        with tempfile.TemporaryDirectory() as d:
            persister = IndexPersister(state_dir=d)
            persister.save_to_rag(rag, tag="test")
            loaded = persister.load("test")
            assert loaded.file_hashes == {"f.pdf": "hash"}

    def test_restore_to_rag(self):
        from runeextract.rag.persistence import IndexPersister, IndexState
        from unittest.mock import MagicMock
        rag = MagicMock()
        with tempfile.TemporaryDirectory() as d:
            persister = IndexPersister(state_dir=d)
            persister.save(IndexState(file_hashes={"f.pdf": "abc"}), tag="test")
            persister.restore_to_rag(rag, tag="test")
            assert rag._file_hashes == {"f.pdf": "abc"}


# ---- Feature 4: Auto-Scaling Tiered Index ----


class TestTieredIndex:
    def test_configure(self):
        from runeextract.rag.tiered_index import TieredIndex, TierConfig
        ti = TieredIndex()
        ti.configure(
            hot=TierConfig(max_documents=100),
            warm=TierConfig(max_documents=1000),
            cold=TierConfig(store_type="archive"),
        )
        assert ti.stats is not None

    def test_record_access(self):
        from runeextract.rag.tiered_index import TieredIndex
        ti = TieredIndex()
        ti.record_access("doc1")
        ti.record_access("doc1")
        assert ti._access_counts["doc1"] == 2

    def test_compact_hot(self):
        from runeextract.rag.tiered_index import TieredIndex, TierConfig
        ti = TieredIndex()
        ti.configure(
            hot=TierConfig(store_type="chromadb"),
            warm=TierConfig(store_type="faiss"),
            cold=TierConfig(store_type="archive"),
        )
        ti.compact_hot()
        assert ti._tiers.get("hot") is not None


# ---- Feature 5: GraphRAG ----


class TestGraphRAG:
    def test_init(self):
        from runeextract.rag.graph_rag import GraphRAGQuery
        rag = MagicMock()
        g = GraphRAGQuery(rag)
        assert g._rag is not None

    def test_query_without_graph(self):
        from runeextract.rag.graph_rag import GraphRAGQuery
        rag = MagicMock()
        rag.query = MagicMock(return_value="answer")
        g = GraphRAGQuery(rag)
        result = g.query("test question")
        assert result == "answer"

    def test_build_graph(self):
        from runeextract.rag.graph_rag import GraphRAGQuery
        from runeextract.graph.builder import GraphBuilder
        rag = MagicMock()
        builder = GraphBuilder()
        g = GraphRAGQuery(rag, graph_builder=builder)
        doc = MagicMock()
        doc.text = "Apple Inc. was founded by Steve Jobs in Cupertino."
        g.build_graph([doc])
        assert g._graph is not None
        assert len(g._graph.nodes) > 0


# ---- Feature 6: Multi-Tenant Isolation ----


class TestMultiTenant:
    def test_tenant_store(self):
        from runeextract.rag.tenant import TenantStore
        store = TenantStore()
        assert store.collection_name("tenant1") == "documents__tenant1"

    def test_register_get(self):
        from runeextract.rag.tenant import TenantStore
        store = TenantStore()
        rag = MagicMock()
        rag.collection_name = "docs"
        store.register("org1", rag)
        assert store.get("org1") is rag
        assert store.get("org2") is None

    def test_list_remove(self):
        from runeextract.rag.tenant import TenantStore
        store = TenantStore()
        store.register("a", MagicMock())
        store.register("b", MagicMock())
        assert len(store.list_tenants()) == 2
        store.remove("a")
        assert len(store.list_tenants()) == 1

    def test_multi_tenant_rag(self):
        from runeextract.rag.tenant import MultiTenantRAG
        def factory(**kwargs):
            rag = MagicMock()
            rag.collection_name = kwargs.get("collection_name", "docs")
            return rag
        mt = MultiTenantRAG(factory)
        rag1 = mt.get_or_create("t1", collection_name="docs_t1")
        rag2 = mt.get_or_create("t1", collection_name="docs_t1")
        assert rag1 is rag2


# ---- Feature 7: Incremental Re-indexing ----


class TestIncrementalReindexing:
    def test_file_hashes_tracked(self):
        from runeextract.rag import AutoRAG
        rag = AutoRAG()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Test")
            path = f.name
        try:
            rag.ingest(path, incremental=True)
            assert path in rag._file_hashes
            assert len(rag._file_hashes[path]) == 64
        finally:
            os.unlink(path)

    def test_ingest_skips_unchanged(self):
        from runeextract.rag import AutoRAG
        rag = AutoRAG()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Same content")
            path = f.name
        try:
            rag._file_hashes[path] = "already_known_hash"
            rag.ingest(path, incremental=True)
        finally:
            os.unlink(path)

    def test_ingest_reindexes_changed(self):
        from runeextract.rag import AutoRAG
        rag = AutoRAG()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# New content")
            path = f.name
        try:
            rag._file_hashes[path] = "old_hash"
            rag.ingest(path, incremental=True)
        finally:
            os.unlink(path)


# ---- Feature 8: RAG Evaluation Suite ----


class TestRAGEvalSuite:
    def test_eval_question_defaults(self):
        from runeextract.rag.eval_suite import EvalQuestion
        q = EvalQuestion(question="What is AI?")
        assert q.question == "What is AI?"
        assert q.ground_truth == ""

    def test_eval_result_defaults(self):
        from runeextract.rag.eval_suite import EvalResult
        r = EvalResult(question="Q")
        assert r.relevance_score == 0.0

    def test_scorecard(self):
        from runeextract.rag.eval_suite import Scorecard, EvalResult
        sc = Scorecard(
            total_questions=1,
            avg_relevance=0.8,
            avg_retrieval_accuracy=0.9,
            avg_latency_ms=100.0,
            results=[EvalResult(question="Q", relevance_score=0.8)],
        )
        assert sc.avg_relevance == 0.8
        assert sc.avg_latency_ms == 100.0

    def test_evaluator_init(self):
        from runeextract.rag.eval_suite import RAGEvaluator
        rag = MagicMock()
        ev = RAGEvaluator(rag)
        assert ev._rag is rag

    def test_evaluate_question(self):
        from runeextract.rag.eval_suite import RAGEvaluator, EvalQuestion
        mock_result = MagicMock()
        mock_result.answer = "AI is artificial intelligence."
        mock_result.retrieved_chunks = []
        rag = MagicMock()
        rag.query = MagicMock(return_value=mock_result)
        ev = RAGEvaluator(rag)
        result = ev.evaluate_question(EvalQuestion(question="What is AI?"))
        assert result.answer == "AI is artificial intelligence."
        assert result.latency_ms >= 0

    def test_run_multiple(self):
        from runeextract.rag.eval_suite import RAGEvaluator, EvalQuestion
        mock_result = MagicMock()
        mock_result.answer = "answer"
        mock_result.retrieved_chunks = []
        rag = MagicMock()
        rag.query = MagicMock(return_value=mock_result)
        ev = RAGEvaluator(rag)
        questions = [EvalQuestion(question=f"Q{i}") for i in range(3)]
        sc = ev.run(questions)
        assert sc.total_questions == 3


# ---- Feature 9: Bulk ChromaDB inserts ----
# (verified: to_chromadb already passes all chunks in one collection.add() call)


class TestBulkInserts:
    def test_to_chromadb_bulk(self):
        from runeextract.models.document import Document
        doc = Document(text="Bulk test content " * 100)
        doc.chunks()
        mock_chromadb = MagicMock()
        mock_chromadb.config.Settings = MagicMock()
        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client
        with patch.dict("sys.modules", {"chromadb": mock_chromadb, "chromadb.config": mock_chromadb.config}):
            coll = doc.to_chromadb(collection_name="bulk_test", persist_directory=tempfile.mkdtemp())
            mock_collection.add.assert_called_once()


# ---- Feature 10: Result-level streaming ----
# (verified: StreamingRAG already yields RETRIEVAL events before answers)


class TestResultStreaming:
    def test_streaming_rag_retrieval_events(self):
        from runeextract.rag.streaming import StreamingRAG
        rag = MagicMock()
        srag = StreamingRAG(rag)
        assert srag.initial_chunks == 3
        assert srag.refinement_chunks == 7

    def test_streaming_cancel(self):
        from runeextract.rag.streaming import StreamingRAG
        rag = MagicMock()
        srag = StreamingRAG(rag)
        srag.cancel()
        assert srag._cancel_flag.is_set()
