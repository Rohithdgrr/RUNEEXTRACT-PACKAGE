"""Tests for DAG pipeline (runeextract.transform)."""

from unittest.mock import patch, MagicMock
import pytest

from runeextract.transform import (
    Pipeline, DagPipeline, PipelineStep, PipelineContext, PipelineResult, run_pipeline,
    ExtractStep, ExtractManyStep, ChunkStep, FilterStep, MapStep,
    AIStep, EmbedStep, StoreStep, LogStep,
    ConditionalStep, ParallelStep, WaitStep,
)
from runeextract.models.document import Document, ChunkingStrategy
from runeextract.transform.steps import _get_ai, _reset_ai


@pytest.fixture(autouse=True)
def _reset_ai_singleton():
    _reset_ai()
    yield
    _reset_ai()


# ---------- Helper ----------

def _make_doc(text: str, source_path: str = "test.txt") -> Document:
    return Document(text=text, source_type="text", source_path=source_path)


# ==================== Pipeline Step base ====================


class TestPipelineStep:
    def test_base_run_raises(self):
        step = PipelineStep("test")
        with pytest.raises(NotImplementedError):
            step.run(PipelineContext())

    def test_repr(self):
        step = PipelineStep("my_step")
        assert "PipelineStep" in repr(step)
        assert "my_step" in repr(step)


# ==================== Pipeline ====================


class TestPipeline:
    def test_empty_pipeline_raises(self):
        with pytest.raises(ValueError, match="at least one step"):
            Pipeline()

    def test_single_step(self):
        class SimpleStep(PipelineStep):
            def run(self, ctx):
                ctx.documents.append(_make_doc("hello"))
                return "ok"

        pipe = Pipeline(SimpleStep("step1"))
        result = pipe.run()
        assert result.steps_run == 1
        assert len(result.documents) == 1
        assert result.documents[0].text == "hello"

    def test_multi_step_order(self):
        class AppendStep(PipelineStep):
            def __init__(self, name, text):
                super().__init__(name)
                self.text = text
            def run(self, ctx):
                ctx.documents.append(_make_doc(self.text))
                return self.text

        pipe = Pipeline(
            AppendStep("a", "first"),
            AppendStep("b", "second"),
        )
        result = pipe.run()
        assert len(result.documents) == 2
        assert result.documents[0].text == "first"
        assert result.documents[1].text == "second"

    def test_step_outputs_stored(self):
        class ReturnStep(PipelineStep):
            def run(self, ctx):
                return 42

        pipe = Pipeline(ReturnStep("magic"))
        result = pipe.run()
        assert result.step_results["magic"] == 42

    def test_stop_on_error(self):
        class FailStep(PipelineStep):
            def run(self, ctx):
                raise ValueError("boom")

        pipe = Pipeline(
            FailStep("fail"),
            ExtractStep("extract", file_path="dummy.pdf"),
        )
        result = pipe.run(stop_on_error=True)
        assert result.steps_run == 0  # first step failed, didn't continue
        assert "fail" in result.errors

    def test_continue_on_error(self):
        class FailStep(PipelineStep):
            def run(self, ctx):
                raise ValueError("boom")

        class OkStep(PipelineStep):
            def run(self, ctx):
                ctx.documents.append(_make_doc("ok"))
                return "ok"

        pipe = Pipeline(FailStep("fail"), OkStep("ok"))
        result = pipe.run(stop_on_error=False)
        assert result.steps_run == 1  # only the ok step ran
        assert "fail" in result.errors

    def test_add_step(self):
        pipe = Pipeline(LogStep("log1"))
        pipe.add_step(LogStep("log2"))
        assert len(pipe._steps) == 2
        pipe.add_step(LogStep("log3"), index=1)
        assert pipe._steps[1].name == "log3"

    def test_run_pipeline_convenience(self):
        class SimpleStep(PipelineStep):
            def run(self, ctx):
                return "done"

        result = run_pipeline(SimpleStep("s"))
        assert isinstance(result, PipelineResult)

    def test_custom_context(self):
        step = LogStep("log")
        ctx = PipelineContext(
            documents=[_make_doc("custom")],
            config={"key": "value"},
        )
        pipe = Pipeline(step)
        result = pipe.run(ctx=ctx)
        assert len(result.documents) == 1


# ==================== Concrete Steps ====================


class TestExtractStep:
    @patch("runeextract.extract")
    def test_extract_single(self, mock_extract):
        mock_extract.return_value = _make_doc("extracted text", "file.pdf")
        step = ExtractStep("extract", file_path="file.pdf", ocr=True)
        ctx = PipelineContext()
        result = step.run(ctx)
        assert result.text == "extracted text"
        assert len(ctx.documents) == 1
        mock_extract.assert_called_with("file.pdf", ocr=True)

    def test_extract_no_path_raises(self):
        step = ExtractStep("extract")
        with pytest.raises(ValueError, match="file_path"):
            step.run(PipelineContext())


class TestExtractManyStep:
    @patch("runeextract.extract")
    def test_extract_many(self, mock_extract):
        mock_extract.side_effect = [
            _make_doc("doc1", "a.txt"),
            _make_doc("doc2", "b.txt"),
        ]
        step = ExtractManyStep("multi", file_paths=["a.txt", "b.txt"])
        ctx = PipelineContext()
        result = step.run(ctx)
        assert len(result) == 2
        assert len(ctx.documents) == 2

    @patch("runeextract.extract")
    def test_extract_many_skips_errors(self, mock_extract):
        mock_extract.side_effect = [
            _make_doc("doc1", "a.txt"),
            ValueError("fail"),
        ]
        step = ExtractManyStep("multi", file_paths=["a.txt", "b.txt"])
        ctx = PipelineContext()
        result = step.run(ctx)
        assert len(result) == 1  # one skipped

    def test_extract_many_no_paths_raises(self):
        step = ExtractManyStep("multi")
        with pytest.raises(ValueError, match="file_paths"):
            step.run(PipelineContext())


class TestChunkStep:
    def test_chunk_all(self):
        doc = _make_doc("word " * 5000)
        step = ChunkStep("chunk", strategy="fixed_size", chunk_size=1000, chunk_overlap=100)
        ctx = PipelineContext(documents=[doc])
        step.run(ctx)
        assert doc._chunks is not None
        assert len(doc._chunks) > 0

    def test_chunk_last_only(self):
        doc1 = _make_doc("a" * 5000)
        doc2 = _make_doc("b" * 5000)
        step = ChunkStep("chunk", target="last", strategy="fixed_size", chunk_size=2000)
        ctx = PipelineContext(documents=[doc1, doc2])
        step.run(ctx)
        assert doc1._chunks is None  # not chunked
        assert doc2._chunks is not None  # chunked

    def test_chunk_empty_docs(self):
        step = ChunkStep("chunk", strategy="fixed_size")
        ctx = PipelineContext(documents=[])
        result = step.run(ctx)
        assert result == []


class TestFilterStep:
    def test_filter_min_length(self):
        docs = [_make_doc("short"), _make_doc("longer text here")]
        step = FilterStep("filter", min_length=10)
        ctx = PipelineContext(documents=docs)
        step.run(ctx)
        assert len(ctx.documents) == 1
        assert ctx.documents[0].text == "longer text here"

    def test_filter_max_length(self):
        docs = [_make_doc("short"), _make_doc("this is a very long text")]
        step = FilterStep("filter", max_length=10)
        ctx = PipelineContext(documents=docs)
        step.run(ctx)
        assert len(ctx.documents) == 1
        assert ctx.documents[0].text == "short"

    def test_filter_predicate(self):
        docs = [_make_doc("keep me"), _make_doc("drop me")]
        step = FilterStep("filter", predicate=lambda d: "keep" in d.text)
        ctx = PipelineContext(documents=docs)
        step.run(ctx)
        assert len(ctx.documents) == 1

    def test_filter_all_dropped(self):
        docs = [_make_doc("a"), _make_doc("b")]
        step = FilterStep("filter", min_length=100)
        ctx = PipelineContext(documents=docs)
        step.run(ctx)
        assert len(ctx.documents) == 0


class TestMapStep:
    def test_map_transform(self):
        doc = _make_doc("original")
        step = MapStep("upper", fn=lambda d: _make_doc(d.text.upper()))
        ctx = PipelineContext(documents=[doc])
        step.run(ctx)
        assert ctx.documents[0].text == "ORIGINAL"

    def test_map_drop_returning_none(self):
        docs = [_make_doc("keep"), _make_doc("drop")]
        step = MapStep("drop", fn=lambda d: d if d.text == "keep" else None)
        ctx = PipelineContext(documents=docs)
        step.run(ctx)
        assert len(ctx.documents) == 1
        assert ctx.documents[0].text == "keep"

    def test_map_no_fn_raises(self):
        step = MapStep("bad")
        with pytest.raises(ValueError, match="fn"):
            step.run(PipelineContext())


class TestAIStep:
    @patch("runeextract.processors.ai.AIProcessor")
    def test_ai_summarize(self, mock_ai_cls):
        mock_ai = MagicMock()
        mock_ai.summarize.return_value = "summary text"
        mock_ai_cls.return_value = mock_ai

        doc = _make_doc("long text here " * 100)
        step = AIStep("ai", action="summarize", max_words=50)
        ctx = PipelineContext(documents=[doc])
        result = step.run(ctx)

        mock_ai.summarize.assert_called_once()
        assert doc.metadata.get("summarize") == "summary text"
        assert result == ["summary text"]

    @patch("runeextract.processors.ai.AIProcessor")
    def test_ai_extract_entities(self, mock_ai_cls):
        mock_ai = MagicMock()
        mock_ai.extract_entities.return_value = [{"type": "person", "name": "Alice"}]
        mock_ai_cls.return_value = mock_ai

        doc = _make_doc("Alice went to Paris.")
        step = AIStep("ai", action="extract_entities")
        ctx = PipelineContext(documents=[doc])
        result = step.run(ctx)

        assert result[0][0]["name"] == "Alice"

    @patch("runeextract.processors.ai.AIProcessor")
    def test_ai_extract_keywords(self, mock_ai_cls):
        mock_ai = MagicMock()
        mock_ai.extract_keywords.return_value = ["ai", "ml"]
        mock_ai_cls.return_value = mock_ai

        doc = _make_doc("AI and ML are related.")
        step = AIStep("ai", action="extract_keywords", top_n=2)
        ctx = PipelineContext(documents=[doc])
        result = step.run(ctx)

        assert result[0] == ["ai", "ml"]

    @patch("runeextract.processors.ai.AIProcessor")
    def test_ai_answer_question(self, mock_ai_cls):
        mock_ai = MagicMock()
        mock_ai.answer_question.return_value = "Paris"
        mock_ai_cls.return_value = mock_ai

        doc = _make_doc("Paris is the capital of France.")
        step = AIStep("ai", action="answer_question", question="What is the capital?")
        ctx = PipelineContext(documents=[doc])
        step.run(ctx)

        mock_ai.answer_question.assert_called_with("What is the capital?", doc.text)

    @patch("runeextract.processors.ai.AIProcessor")
    def test_ai_unknown_action_raises(self, mock_ai_cls):
        mock_ai = MagicMock()
        mock_ai_cls.return_value = mock_ai
        step = AIStep("bad", action="nonexistent")
        with pytest.raises(ValueError, match="Unknown AIStep action"):
            step.run(PipelineContext(documents=[_make_doc("test")]))


class TestEmbedStep:
    @patch("runeextract.processors.ai.AIProcessor")
    def test_embed(self, mock_ai_cls):
        mock_ai = MagicMock()
        mock_ai.embed.return_value = [[0.1, 0.2, 0.3]]
        mock_ai_cls.return_value = mock_ai

        doc = _make_doc("embed me")
        step = EmbedStep("embed")
        ctx = PipelineContext(documents=[doc])
        result = step.run(ctx)

        assert "embedding" in doc.metadata
        assert result[0] == [[0.1, 0.2, 0.3]]


class TestStoreStep:
    def test_store_json_output(self, tmp_path):
        doc = _make_doc("storable text")
        doc._chunks = []  # simulate empty chunk list
        step = StoreStep("store", output_dir=str(tmp_path))
        ctx = PipelineContext(documents=[doc])
        result = step.run(ctx)

        assert result["document_count"] == 1
        out_file = tmp_path / "pipeline_output.json"
        assert out_file.exists()

    def test_store_jsonl_output(self, tmp_path):
        doc = _make_doc("storable")
        step = StoreStep("store", output_dir=str(tmp_path), format="jsonl")
        ctx = PipelineContext(documents=[doc])
        step.run(ctx)

        out_file = tmp_path / "pipeline_output.jsonl"
        assert out_file.exists()
        content = out_file.read_text()
        assert "test.txt" in content

    def test_store_no_output_dir(self):
        doc = _make_doc("no dir")
        step = StoreStep("store")
        ctx = PipelineContext(documents=[doc])
        result = step.run(ctx)
        assert result["document_count"] == 1


class TestLogStep:
    def test_log_step(self):
        step = LogStep("log", message="Got {doc_count} docs")
        ctx = PipelineContext(documents=[_make_doc("a"), _make_doc("b")])
        result = step.run(ctx)
        assert "Got 2 docs" in result


# ==================== PipelineStep retry & skip ====================


class TestPipelineStepRetry:
    def test_retry_eventually_succeeds(self):
        """Step fails twice then succeeds on third attempt."""
        attempt_count = [0]

        class FlakyStep(PipelineStep):
            def run(self, ctx):
                attempt_count[0] += 1
                if attempt_count[0] < 3:
                    raise ValueError("not yet")
                return "success"

        step = FlakyStep("flaky", retry_count=3, retry_delay=0.01)
        ctx = PipelineContext()
        result = step._execute(ctx)
        assert result == "success"
        assert attempt_count[0] == 3

    def test_retry_exhausted_raises(self):
        attempt_count = [0]

        class AlwaysFailStep(PipelineStep):
            def run(self, ctx):
                attempt_count[0] += 1
                raise ValueError("always fail")

        step = AlwaysFailStep("fail", retry_count=2, retry_delay=0.01)
        ctx = PipelineContext()
        with pytest.raises(ValueError, match="always fail"):
            step._execute(ctx)
        assert attempt_count[0] == 3  # initial + 2 retries

    def test_skip_if_skips_step(self):
        class SimpleStep(PipelineStep):
            def run(self, ctx):
                ctx.documents.append(_make_doc("ran"))
                return "ran"

        step = SimpleStep("skipme", skip_if=lambda ctx: True)
        ctx = PipelineContext()
        result = step._execute(ctx)
        assert result is None
        assert len(ctx.documents) == 0  # step didn't run

    def test_skip_if_false_runs_normally(self):
        class SimpleStep(PipelineStep):
            def run(self, ctx):
                ctx.documents.append(_make_doc("ran"))
                return "ran"

        step = SimpleStep("dontskip", skip_if=lambda ctx: False)
        ctx = PipelineContext()
        result = step._execute(ctx)
        assert result == "ran"
        assert len(ctx.documents) == 1


# ==================== DagPipeline ====================


class TestDagPipeline:
    def test_single_step(self):
        class SimpleStep(PipelineStep):
            def run(self, ctx):
                ctx.documents.append(_make_doc("hello"))
                return "ok"

        dag = DagPipeline()
        dag.add_step(SimpleStep("step1"))
        result = dag.run()
        assert result.steps_run == 1
        assert len(result.documents) == 1

    def test_dependency_order(self):
        class AppendStep(PipelineStep):
            def __init__(self, name, text):
                super().__init__(name)
                self.text = text
            def run(self, ctx):
                ctx.documents.append(_make_doc(self.text))
                return self.text

        dag = DagPipeline()
        dag.add_step(AppendStep("first", "alpha"))
        dag.add_step(AppendStep("second", "beta"), depends_on="first")
        dag.add_step(AppendStep("third", "gamma"), depends_on="second")
        result = dag.run()
        assert [d.text for d in result.documents] == ["alpha", "beta", "gamma"]

    def test_multiple_dependencies(self):
        class AppendStep(PipelineStep):
            def __init__(self, name, text):
                super().__init__(name)
                self.text = text
            def run(self, ctx):
                ctx.documents.append(_make_doc(self.text))
                return self.text

        dag = DagPipeline()
        dag.add_step(AppendStep("a", "first"))
        dag.add_step(AppendStep("b", "second"))
        dag.add_step(AppendStep("c", "third"), depends_on=["a", "b"])
        result = dag.run()
        assert len(result.documents) == 3
        assert result.documents[2].text == "third"

    def test_cycle_detection(self):
        class SimpleStep(PipelineStep):
            def run(self, ctx):
                return None

        dag = DagPipeline()
        dag.add_step(SimpleStep("a"), depends_on="b")
        dag.add_step(SimpleStep("b"), depends_on="a")
        with pytest.raises(ValueError, match="Cycle detected"):
            dag.run()

    def test_stop_on_error(self):
        class FailStep(PipelineStep):
            def run(self, ctx):
                raise ValueError("boom")

        dag = DagPipeline()
        dag.add_step(FailStep("fail"))
        dag.add_step(LogStep("log", message="should not run"))
        result = dag.run(stop_on_error=True)
        assert result.steps_run == 0
        assert "fail" in result.errors
        assert "log" not in result.errors

    def test_continue_on_error(self):
        class FailStep(PipelineStep):
            def run(self, ctx):
                raise ValueError("boom")

        class OkStep(PipelineStep):
            def run(self, ctx):
                ctx.documents.append(_make_doc("ok"))
                return "ok"

        dag = DagPipeline()
        dag.add_step(FailStep("fail"))
        dag.add_step(OkStep("ok"))
        result = dag.run(stop_on_error=False)
        assert result.steps_run == 1
        assert "fail" in result.errors
        assert len(result.documents) == 1

    def test_parallel_execution(self):
        class SlowStep(PipelineStep):
            def __init__(self, name, delay=0.1):
                super().__init__(name)
                self.delay = delay
            def run(self, ctx):
                import time
                time.sleep(self.delay)
                ctx.documents.append(_make_doc(self.name))
                return self.name

        dag = DagPipeline(parallel=True, max_workers=2)
        dag.add_step(SlowStep("fast", delay=0.05))
        dag.add_step(SlowStep("slow", delay=0.2))
        result = dag.run(stop_on_error=True)
        assert len(result.documents) == 2

    def test_from_pipeline(self):
        pipe = Pipeline(LogStep("log1"), LogStep("log2"))
        dag = DagPipeline.from_pipeline(pipe)
        assert len(dag._steps) == 2
        assert dag._steps[0].name == "log1"

    def test_init_with_tuples(self):
        class SimpleStep(PipelineStep):
            def run(self, ctx):
                return "ok"
        dag = DagPipeline(
            (SimpleStep("a"), None),
            (SimpleStep("b"), "a"),
        )
        result = dag.run()
        assert result.steps_run == 2


# ==================== Checkpointing ====================


class TestCheckpoint:
    def test_context_save_and_load(self, tmp_path):
        ctx = PipelineContext(
            documents=[_make_doc("saved text")],
            config={"key": "val"},
            metadata={"version": 1},
        )
        path = str(tmp_path / "checkpoint.json")
        ctx.save(path)
        loaded = PipelineContext.load(path)
        assert len(loaded.documents) == 1
        assert loaded.documents[0].text == "saved text"
        assert loaded.config["key"] == "val"
        assert loaded.metadata["version"] == 1

    def test_context_load_missing(self):
        with pytest.raises(FileNotFoundError):
            PipelineContext.load("/nonexistent/checkpoint.json")

    def test_dag_resume(self, tmp_path):
        class CountStep(PipelineStep):
            def __init__(self, name, counter):
                super().__init__(name)
                self.counter = counter
            def run(self, ctx):
                self.counter[0] += 1
                ctx.documents.append(_make_doc(self.name))
                return self.name

        counter = [0]
        cp_path = str(tmp_path / "resume.json")

        # First run
        dag = DagPipeline()
        dag.add_step(CountStep("s1", counter))
        dag.add_step(CountStep("s2", counter), depends_on="s1")
        ctx = PipelineContext()
        result = dag.run(ctx=ctx, checkpoint_path=cp_path)
        assert result.steps_run == 2
        assert counter[0] == 2

        # Second run with resume — docs restored, no new execution
        counter2 = [0]
        dag2 = DagPipeline()
        dag2.add_step(CountStep("s1", counter2))
        dag2.add_step(CountStep("s2", counter2), depends_on="s1")
        result2 = dag2.run(resume_from=cp_path)
        assert result2.steps_run == 0  # all steps already done in checkpoint
        assert len(result2.documents) == 2  # documents restored from checkpoint
        assert counter2[0] == 0  # no step actually executed


# ==================== ConditionalStep / ParallelStep / WaitStep ====================


class TestSpecialSteps:
    def test_conditional_true_branch(self):
        condition = lambda ctx: len(ctx.documents) > 0
        if_step = LogStep("if_branch", message="if ran")
        else_step = LogStep("else_branch", message="else ran")
        step = ConditionalStep("cond", condition, if_step, else_step)
        ctx = PipelineContext(documents=[_make_doc("doc")])
        result = step._execute(ctx)
        assert "if" in str(result)

    def test_conditional_false_branch(self):
        condition = lambda ctx: len(ctx.documents) > 10
        if_step = LogStep("if_branch", message="if ran")
        else_step = LogStep("else_branch", message="else ran")
        step = ConditionalStep("cond", condition, if_step, else_step)
        ctx = PipelineContext(documents=[_make_doc("doc")])
        result = step._execute(ctx)
        assert "else" in str(result)

    def test_conditional_no_else(self):
        condition = lambda ctx: False
        if_step = LogStep("if_branch", message="should not run")
        step = ConditionalStep("cond", condition, if_step)
        ctx = PipelineContext()
        result = step._execute(ctx)
        assert result is None

    def test_wait_step(self):
        step = WaitStep("wait", delay=0.01)
        start = __import__("time").time()
        result = step._execute(PipelineContext())
        elapsed = __import__("time").time() - start
        assert elapsed >= 0.01
        assert "Waited" in result

    def test_parallel_step(self):
        class SimpleStep(PipelineStep):
            def __init__(self, name, delay=0.05):
                super().__init__(name)
                self.delay = delay
            def run(self, ctx):
                import time
                time.sleep(self.delay)
                ctx.documents.append(_make_doc(self.name))
                return self.name

        step = ParallelStep("parallel", SimpleStep("p1", 0.05), SimpleStep("p2", 0.1), max_workers=2)
        ctx = PipelineContext()
        results = step._execute(ctx)
        assert "p1" in results
        assert "p2" in results
        assert len(ctx.documents) == 2
