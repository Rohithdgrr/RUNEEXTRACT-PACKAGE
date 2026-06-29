"""Tests for RAG maintenance modules: versioning, budget, dedup, vision, repair."""

import os
import time
import tempfile
from unittest.mock import MagicMock, patch, Mock
from dataclasses import dataclass


# ── Index Versioning ──

class TestIndexVersioning:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_snapshot_creates_entry(self):
        from runeextract.rag.versioning import IndexVersioning
        ver = IndexVersioning(persist_directory=self.tmpdir)
        snap = ver.snapshot(embedding_model="test:model", chunking="semantic", chunk_size=500, tag="v1")
        assert snap.id >= 1
        assert snap.embedding_model == "test:model"
        assert snap.chunking == "semantic"
        assert snap.chunk_size == 500
        assert snap.tag == "v1"
        assert snap.total_files == 0
        ver.close()

    def test_list_snapshots_returns_recent(self):
        from runeextract.rag.versioning import IndexVersioning
        ver = IndexVersioning(persist_directory=self.tmpdir)
        ver.snapshot(tag="a")
        time.sleep(0.01)
        ver.snapshot(tag="b")
        snaps = ver.list_snapshots(limit=5)
        assert len(snaps) == 2
        assert snaps[0].tag == "b"
        assert snaps[1].tag == "a"
        ver.close()

    def test_get_snapshot_by_id(self):
        from runeextract.rag.versioning import IndexVersioning
        ver = IndexVersioning(persist_directory=self.tmpdir)
        snap = ver.snapshot(tag="test")
        retrieved = ver.get_snapshot(snap.id)
        assert retrieved is not None
        assert retrieved.id == snap.id
        assert retrieved.tag == "test"
        ver.close()

    def test_get_snapshot_missing(self):
        from runeextract.rag.versioning import IndexVersioning
        ver = IndexVersioning(persist_directory=self.tmpdir)
        assert ver.get_snapshot(9999) is None
        ver.close()

    def test_changelog_empty_initially(self):
        from runeextract.rag.versioning import IndexVersioning
        ver = IndexVersioning(persist_directory=self.tmpdir)
        report = ver.changelog(days=30)
        assert report.entries == []
        ver.close()

    def test_changelog_after_multiple_snapshots(self):
        from runeextract.rag.versioning import IndexVersioning
        ver = IndexVersioning(persist_directory=self.tmpdir)
        ver.snapshot(tag="first")
        ver.snapshot(tag="second")
        report = ver.changelog(days=30)
        assert report.total_snapshots >= 2
        assert report.added >= 0
        assert report.removed >= 0
        ver.close()

    def test_context_manager(self):
        from runeextract.rag.versioning import IndexVersioning
        with IndexVersioning(persist_directory=self.tmpdir) as ver:
            snap = ver.snapshot(tag="ctx")
            assert snap.id >= 1

    def test_snapshot_info_created_str(self):
        from runeextract.rag.versioning import SnapshotInfo
        info = SnapshotInfo(id=1, created_at=1000000.0, tag="t",
                            embedding_model="m", chunking="c", chunk_size=100,
                            total_files=2, total_chunks=10)
        assert "1970" in info.created_str

    def test_snapshot_prints_with_print_method(self):
        from runeextract.rag.versioning import ChangelogReport, ChangelogEntry
        report = ChangelogReport(
            entries=[ChangelogEntry(action="added", path="doc.md")],
            added=1, removed=0, modified=0, total_snapshots=1,
        )
        text = report.print()
        assert "+" in text
        assert "doc.md" in text


class TestIndexVersioningEdgeCases:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_db_path_custom(self):
        from runeextract.rag.versioning import IndexVersioning
        db = os.path.join(self.tmpdir, "custom.db")
        with IndexVersioning(persist_directory=self.tmpdir, db_path=db) as ver:
            snap = ver.snapshot(tag="custom")
            assert snap.id >= 1
            assert os.path.exists(db)

    def test_multiple_snapshots_preserve_order(self):
        from runeextract.rag.versioning import IndexVersioning
        ver = IndexVersioning(persist_directory=self.tmpdir)
        ids = []
        for i in range(5):
            s = ver.snapshot(tag=f"v{i}")
            ids.append(s.id)
        assert ids == sorted(ids)
        ver.close()

    def test_list_snapshots_limit(self):
        from runeextract.rag.versioning import IndexVersioning
        ver = IndexVersioning(persist_directory=self.tmpdir)
        for i in range(10):
            ver.snapshot(tag=f"v{i}")
        assert len(ver.list_snapshots(limit=3)) == 3
        ver.close()

    def test_changelog_prints_with_method(self):
        from runeextract.rag.versioning import ChangelogReport, ChangelogEntry
        report = ChangelogReport()
        report.print()
        report.added = 2
        report.entries = [ChangelogEntry("added", "f1"), ChangelogEntry("removed", "f2")]
        text = report.print()
        assert "f1" in text
        assert "f2" in text


# ── Budget Manager ──

class TestBudgetManager:
    def test_can_query_no_limits(self):
        from runeextract.rag.budget import BudgetManager
        bm = BudgetManager()
        allowed, reason = bm.can_query()
        assert allowed is True
        assert reason is None

    def test_cost_per_query_exceeded(self):
        from runeextract.rag.budget import BudgetManager, BudgetConfig
        config = BudgetConfig(cost_per_query=0.01)
        bm = BudgetManager(config=config)
        allowed, reason = bm.can_query(estimated_cost=0.02)
        assert allowed is False
        assert "per-query limit" in (reason or "")

    def test_cost_per_day_exceeded(self):
        from runeextract.rag.budget import BudgetManager, BudgetConfig
        config = BudgetConfig(cost_per_day=0.05)
        bm = BudgetManager(config=config)
        allowed, reason = bm.can_query(estimated_cost=0.06)
        assert allowed is False
        assert "Daily cost" in (reason or "")

    def test_rate_limit_exceeded(self):
        from runeextract.rag.budget import BudgetManager, BudgetConfig
        config = BudgetConfig(max_queries_per_minute=2)
        bm = BudgetManager(config=config)
        bm.record()
        bm.record()
        # 3rd query should be rate-limited
        allowed, reason = bm.can_query()
        assert allowed is False
        assert "Rate limit" in (reason or "")

    def test_record_tracks_cost(self):
        from runeextract.rag.budget import BudgetManager
        bm = BudgetManager()
        bm.record(cost=0.01, latency_ms=100, tokens=50)
        state = bm.get_state()
        assert state.total_cost == 0.01
        assert state.daily_cost == 0.01
        assert state.query_count == 1
        assert len(state.latency_ms) == 1

    def test_degradation_flags_level0(self):
        from runeextract.rag.budget import BudgetManager
        bm = BudgetManager()
        flags = bm.degradation_flags()
        assert flags["skip_reranker"] is False
        assert flags["cache_only"] is False

    def test_degradation_flags_level3(self):
        from runeextract.rag.budget import BudgetManager, BudgetConfig
        config = BudgetConfig(cost_per_day=0.01, on_exceeded="degrade")
        bm = BudgetManager(config=config)
        bm.record(cost=0.02)
        flags = bm.degradation_flags()
        # at least level 1
        assert flags["skip_reranker"] is True

    def test_degrade_response_for_refused(self):
        from runeextract.rag.budget import BudgetManager
        bm = BudgetManager()
        bm._state.degradation_level = 3
        resp = bm.degrade_response("test")
        assert "Budget limit" in resp

    def test_config_from_env(self):
        from runeextract.rag.budget import BudgetConfig
        import os
        os.environ["RUNEEXTRACT_BUDGET_COST_PER_QUERY"] = "0.05"
        os.environ["RUNEEXTRACT_BUDGET_LATENCY_P95_MS"] = "1000"
        try:
            config = BudgetConfig.from_env()
            assert config.cost_per_query == 0.05
            assert config.latency_p95_ms == 1000.0
        finally:
            del os.environ["RUNEEXTRACT_BUDGET_COST_PER_QUERY"]
            del os.environ["RUNEEXTRACT_BUDGET_LATENCY_P95_MS"]

    def test_daily_reset(self):
        from runeextract.rag.budget import BudgetManager, BudgetConfig
        config = BudgetConfig(cost_per_day=1.0)
        bm = BudgetManager(config=config)
        bm.record(cost=0.9)
        # Simulate next day's first query
        bm._state.daily_cost = 0.0
        bm._state.last_reset_day = "2000-01-01"
        allowed, _ = bm.can_query(estimated_cost=0.9)
        assert allowed is True


class TestBudgetExceededError:
    def test_exception_raise_action(self):
        from runeextract.rag.budget import BudgetManager, BudgetConfig, BudgetExceededError
        import pytest
        config = BudgetConfig(cost_per_query=0.01, on_exceeded="raise")
        bm = BudgetManager(config=config)
        with pytest.raises(BudgetExceededError):
            bm.record(cost=0.02)


class TestBudgetWebhook:
    def test_webhook_called_on_degrade(self):
        from runeextract.rag.budget import BudgetManager, BudgetConfig
        calls = []

        def webhook(msg):
            calls.append(msg)

        config = BudgetConfig(cost_per_day=0.01, on_exceeded="degrade")
        bm = BudgetManager(config=config, webhook=webhook)
        bm.record(cost=0.02)
        assert len(calls) > 0


# ── Dedup Engine ──

class TestDedupEngine:
    def test_find_duplicates_empty(self):
        from runeextract.rag.dedup_engine import DedupEngine
        engine = DedupEngine(strategy="minhash", threshold=0.99)
        report = engine.find_duplicates([])
        assert report.total_duplicates == 0
        assert report.groups == []

    def test_find_duplicates_no_duplicates(self):
        from runeextract.rag.dedup_engine import DedupEngine
        engine = DedupEngine(strategy="minhash", threshold=0.99)
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Python is a programming language.",
            "Machine learning is transforming industries.",
        ]
        report = engine.find_duplicates(texts)
        assert report.total_chunks_scanned == 3

    def test_find_duplicates_with_source_map(self):
        from runeextract.rag.dedup_engine import DedupEngine
        engine = DedupEngine(strategy="minhash", threshold=0.5)
        texts = [
            "This is a test document with some content.",
            "This is a test document with some content.",  # identical
            "Completely different text here.",
        ]
        source_map = {0: "doc1.txt", 1: "doc2.txt", 2: "doc3.txt"}
        report = engine.find_duplicates(texts, source_map=source_map)
        assert report.total_chunks_scanned == 3

    def test_dedup_report_print(self):
        from runeextract.rag.dedup_engine import DedupReport, DuplicateGroup
        report = DedupReport(
            groups=[DuplicateGroup(chunks=[{"index": 0, "text": "hi"}],
                                   source_files=["a.txt"],
                                   similarity=0.95,
                                   representative="hi")],
            total_duplicates=1,
            total_chunks_scanned=10,
            savings_estimate=1,
        )
        text = report.print()
        assert "95" in text
        assert "a.txt" in text

    def test_invalid_strategy(self):
        from runeextract.rag.dedup_engine import DedupEngine
        import pytest
        with pytest.raises(ValueError, match="Unknown strategy"):
            DedupEngine(strategy="invalid")


# ── Vision Integration ──

class TestAIProcessorVisionCapability:
    def test_has_vision_openai_vision_model(self):
        from runeextract.processors.ai import AIProcessor
        ai = AIProcessor(api_key="sk-test", provider="openai", model="gpt-4o")
        assert ai.has_vision is True

    def test_has_vision_openai_text_model(self):
        from runeextract.processors.ai import AIProcessor
        ai = AIProcessor(api_key="sk-test", provider="openai", model="gpt-3.5-turbo")
        assert ai.has_vision is False

    def test_has_vision_anthropic_sonnet(self):
        from runeextract.processors.ai import AIProcessor
        ai = AIProcessor(api_key="sk-ant-test", provider="anthropic", model="claude-3-5-sonnet-20241022")
        assert ai.has_vision is True

    def test_has_vision_gemini_pro(self):
        from runeextract.processors.ai import AIProcessor
        ai = AIProcessor(api_key="ai-test", provider="gemini", model="gemini-1.5-pro")
        assert ai.has_vision is True

    def test_describe_image_uses_vision_call(self):
        from runeextract.processors.ai import AIProcessor
        with patch("runeextract.processors.providers.vision_call") as mock_vc:
            mock_vc.return_value = "A test image description"
            ai = AIProcessor(api_key="sk-test", provider="openai", model="gpt-4o")
            result = ai.describe_image(b"fake_image_bytes", image_format="png",
                                       prompt="What is in this image?")
            assert result == "A test image description"
            mock_vc.assert_called_once()
            call_kwargs = mock_vc.call_args.kwargs
            assert call_kwargs["system"] is not None
            assert call_kwargs["user"] == "What is in this image?"
            assert len(call_kwargs["images"]) == 1
            img_data, img_fmt = call_kwargs["images"][0]
            assert img_data == b"fake_image_bytes"
            assert img_fmt == "png"

    def test_analyze_images_empty(self):
        from runeextract.processors.ai import AIProcessor
        ai = AIProcessor(api_key="sk-test", provider="openai", model="gpt-4o")
        result = ai.analyze_images([])
        assert result == []

    def test_analyze_images_single(self):
        from runeextract.processors.ai import AIProcessor
        with patch("runeextract.processors.providers.vision_call") as mock_vc:
            mock_vc.return_value = "Analyzed"
            ai = AIProcessor(api_key="sk-test", provider="openai", model="gpt-4o")
            results = ai.analyze_images([(b"img1", "png")], prompt="Analyze these")
            assert results == ["Analyzed"]
            mock_vc.assert_called_once()
            assert len(mock_vc.call_args.kwargs["images"]) == 1


class TestVisionCallOpenAICompat:
    def test_supports_vision_openai(self):
        from runeextract.processors.providers.openai_compat import supports_vision
        assert supports_vision("gpt-4o") is True
        assert supports_vision("gpt-4o-mini") is True
        assert supports_vision("gpt-3.5-turbo") is False
        assert supports_vision("gpt-4-turbo") is True


class TestVisionCallAnthropic:
    def test_supports_vision_anthropic(self):
        from runeextract.processors.providers.anthropic import supports_vision
        assert supports_vision("claude-3-5-sonnet-20241022") is True
        assert supports_vision("claude-2") is False


class TestVisionCallGemini:
    def test_supports_vision_gemini(self):
        from runeextract.processors.providers.gemini import supports_vision
        assert supports_vision("gemini-1.5-pro") is True
        assert supports_vision("gemini-1.5-flash") is True
        assert supports_vision("text-bison") is False


class TestProvidersSupportsVision:
    def test_known_vision_patterns(self):
        from runeextract.processors.providers import supports_vision
        assert supports_vision("openai", "gpt-4o") is True
        assert supports_vision("openai", "gpt-4-vision-preview") is True
        assert supports_vision("openai", "gpt-3.5-turbo") is False

    def test_unknown_provider_fallback(self):
        from runeextract.processors.providers import supports_vision
        import pytest
        with pytest.raises(ValueError, match="Unknown provider"):
            supports_vision("nonexistent", "some-model")


# ── KnowledgeBase Self-Healing ──

class TestKnowledgeBaseRepair:
    @patch("runeextract.cli.doctor.run_doctor")
    def test_repair_handles_empty_index(self, mock_doctor):
        from runeextract.cli.doctor import DoctorReport
        mock_doctor.return_value = DoctorReport(index_path="", diagnostics=[])
        import sys
        mock_chromadb = MagicMock()
        mock_chromadb.config.Settings = MagicMock()
        mock_chromadb.PersistentClient.return_value.list_collections.return_value = []
        with patch.dict("sys.modules", {"chromadb": mock_chromadb, "chromadb.config": mock_chromadb.config}):
            from runeextract.rag.knowledge_base import KnowledgeBase
            kb = KnowledgeBase("./test_src", persist_directory="./test_db")
            kb._is_built = True
            actions = kb.repair(dry_run=True)
            assert actions == []

    @patch("runeextract.cli.doctor.run_doctor")
    def test_repair_orphan_chunks_dry_run(self, mock_doctor):
        from runeextract.cli.doctor import DoctorReport
        mock_doctor.return_value = DoctorReport(index_path="", diagnostics=[])
        import sys
        mock_chromadb = MagicMock()
        mock_chromadb.config.Settings = MagicMock()
        mock_client = MagicMock()
        mock_col = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.list_collections.return_value = [mock_col]
        mock_col.get.return_value = {
            "documents": ["content"],
            "metadatas": [{"source": "/nonexistent/doc.txt", "chunk_id": "c1"}],
            "embeddings": [[0.1] * 384],
        }
        with patch.dict("sys.modules", {"chromadb": mock_chromadb, "chromadb.config": mock_chromadb.config}), \
             patch("os.path.exists", return_value=False):
            from runeextract.rag.knowledge_base import KnowledgeBase
            kb = KnowledgeBase("./test_src", persist_directory="./test_db")
            kb._is_built = True
            actions = kb.repair(dry_run=True)
            assert any("orphan" in a.lower() for a in actions)

    @patch("runeextract.cli.doctor.run_doctor")
    def test_repair_tiny_chunks(self, mock_doctor):
        from runeextract.cli.doctor import DoctorReport
        mock_doctor.return_value = DoctorReport(index_path="", diagnostics=[])
        import sys
        mock_chromadb = MagicMock()
        mock_chromadb.config.Settings = MagicMock()
        mock_client = MagicMock()
        mock_col = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.list_collections.return_value = [mock_col]
        mock_col.get.return_value = {
            "documents": ["tiny"],
            "metadatas": [{"source": "/real/doc.txt", "chunk_id": "c1", "source_type": "txt", "document_id": "d1"}],
            "embeddings": [[0.1] * 384],
        }
        with patch.dict("sys.modules", {"chromadb": mock_chromadb, "chromadb.config": mock_chromadb.config}), \
             patch("os.path.exists", return_value=True):
            from runeextract.rag.knowledge_base import KnowledgeBase
            kb = KnowledgeBase("./test_src", persist_directory="./test_db")
            kb._is_built = True
            actions = kb.repair(dry_run=True)
            assert any("tiny" in a.lower() for a in actions)


# ── Top-Level Exports ──

class TestTopLevelExports:
    def test_knowledge_base_importable(self):
        from runeextract import KnowledgeBase
        assert callable(KnowledgeBase)

    def test_index_versioning_importable(self):
        from runeextract import IndexVersioning
        assert callable(IndexVersioning)

    def test_budget_manager_importable(self):
        from runeextract import BudgetManager
        assert callable(BudgetManager)

    def test_dedup_engine_importable(self):
        from runeextract import DedupEngine
        assert callable(DedupEngine)

    def test_rag_module_exports(self):
        from runeextract.rag import IndexVersioning, BudgetManager, BudgetConfig, BudgetExceededError
        from runeextract.rag import DedupEngine, DedupReport
        assert callable(IndexVersioning)
        assert callable(BudgetManager)
        assert callable(DedupEngine)

    def test_default_dedup_strategy(self):
        from runeextract.rag.dedup_engine import DedupEngine
        engine = DedupEngine()
        assert engine._strategy == "minhash"
        assert engine._threshold == 0.85


# ── BudgetManager State ──

class TestBudgetState:
    def test_state_defaults(self):
        from runeextract.rag.budget import BudgetState
        state = BudgetState()
        assert state.total_cost == 0.0
        assert state.query_count == 0
        assert state.degradation_level == 0


# ── SnapshotInfo ──

class TestSnapshotInfoEdge:
    def test_snapshot_info_defaults(self):
        from runeextract.rag.versioning import SnapshotInfo
        info = SnapshotInfo(id=0, created_at=0.0, tag=None, embedding_model=None, chunking=None, chunk_size=None)
        assert info.total_files == 0
        assert info.total_chunks == 0
