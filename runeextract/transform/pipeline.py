"""Pipeline DAG — define and execute document processing workflows.

Supports sequential ``Pipeline`` (backward compatible), dependency-resolved
``DagPipeline`` with topological sort, per-step retry, conditional branching,
checkpointing, and parallel execution of independent branches.
"""

import json
import logging
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union

from runeextract.models.document import Document

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ──────────────────────────────────────────────
#  Context & Result (unchanged API)
# ──────────────────────────────────────────────


@dataclass
class PipelineContext:
    """Shared context passed through all pipeline steps.

    Attributes:
        documents: List of documents accumulated during processing.
        step_outputs: Mapping of step name → its return value.
        config: Arbitrary configuration dict shared by all steps.
        metadata: Pipeline-level metadata (timing, counts, etc.).
    """
    documents: List[Document] = field(default_factory=list)
    step_outputs: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── checkpointing ──

    def save(self, path: str) -> str:
        """Save the current context as a JSON checkpoint file.

        Returns the path written to for chaining.
        """
        data = {
            "documents": [
                {"text": d.text, "source_path": d.source_path, "source_type": d.source_type, "metadata": d.metadata}
                for d in self.documents
            ],
            "step_outputs": {k: str(v) for k, v in self.step_outputs.items()},
            "config": dict(self.config),
            "metadata": dict(self.metadata),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Checkpoint saved to %s", path)
        return path

    @classmethod
    def load(cls, path: str) -> "PipelineContext":
        """Load a context from a JSON checkpoint file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        with open(path) as f:
            data = json.load(f)
        docs = [
            Document(text=d.get("text", ""), source_path=d.get("source_path"), source_type=d.get("source_type", ""), metadata=d.get("metadata", {}))
            for d in data.get("documents", [])
        ]
        ctx = cls(
            documents=docs,
            step_outputs=data.get("step_outputs", {}),
            config=data.get("config", {}),
            metadata=data.get("metadata", {}),
        )
        logger.info("Checkpoint loaded from %s", path)
        return ctx


@dataclass
class PipelineResult:
    """Result of a completed pipeline run."""
    documents: List[Document] = field(default_factory=list)
    step_results: Dict[str, Any] = field(default_factory=dict)
    total_time: float = 0.0
    steps_run: int = 0
    errors: Dict[str, str] = field(default_factory=dict)


# ──────────────────────────────────────────────
#  PipelineStep — retry + skip support
# ──────────────────────────────────────────────


class PipelineStep:
    """Base class for a single pipeline processing step.

    Subclasses must implement ``run(self, ctx: PipelineContext)``.

    Args:
        name: Unique step name.
        retry_count: Number of additional attempts on failure (default 0).
        retry_delay: Seconds to wait between retries (default 1.0).
        skip_if: Optional predicate — if it returns True, the step is
            skipped (output is set to None).
    """

    def __init__(
        self,
        name: str,
        retry_count: int = 0,
        retry_delay: float = 1.0,
        skip_if: Optional[Callable[[PipelineContext], bool]] = None,
        **kwargs,
    ):
        self.name = name
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.skip_if = skip_if
        self._kwargs = kwargs

    def run(self, ctx: PipelineContext) -> Any:
        """Execute this step.

        Args:
            ctx: Current pipeline context with documents and previous outputs.

        Returns:
            Arbitrary data stored in ``context.step_outputs[self.name]``.
        """
        raise NotImplementedError

    def _execute(self, ctx: PipelineContext) -> Any:
        """Run the step with retry and skip-if support.

        Called by the pipeline runner.  Subclasses should normally
        override ``run()`` instead.
        """
        if self.skip_if and self.skip_if(ctx):
            logger.info("Step %s skipped (skip_if returned True)", self.name)
            return None

        last_exc = None
        attempts = 1 + self.retry_count
        for attempt in range(1, attempts + 1):
            try:
                return self.run(ctx)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Step %s attempt %d/%d failed: %s",
                    self.name, attempt, attempts, exc,
                )
                if attempt < attempts:
                    time.sleep(self.retry_delay)
        raise last_exc  # type: ignore[misc]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ──────────────────────────────────────────────
#  Sequential Pipeline (backward compatible)
# ──────────────────────────────────────────────


class Pipeline:
    """A composable DAG of document processing steps.

    Steps are executed in declaration order. Each step receives the
    current ``PipelineContext`` and can read/write documents and outputs.

    Usage::

        from runeextract.transform import Pipeline, ExtractStep, ChunkStep

        pipe = Pipeline(
            ExtractStep("extract", file_path="doc.pdf"),
            ChunkStep("chunk", strategy="sentence_window"),
        )
        result = pipe.run()
        # result.documents → list of extracted and chunked documents
    """

    @staticmethod
    def extract(file_path: str, **kwargs):
        """Convenience: extract a single file into a Document.

        Usage::

            doc = Pipeline.extract("doc.pdf", ocr=True)
        """
        from runeextract import extract as _extract
        return _extract(file_path, **kwargs)

    def __init__(self, *steps: PipelineStep):
        if not steps:
            raise ValueError("Pipeline requires at least one step")
        self._steps = list(steps)

    def add_step(self, step: PipelineStep, index: Optional[int] = None) -> "Pipeline":
        if index is None:
            self._steps.append(step)
        else:
            self._steps.insert(index, step)
        return self

    def run(
        self,
        ctx: Optional[PipelineContext] = None,
        stop_on_error: bool = True,
    ) -> PipelineResult:
        context = ctx or PipelineContext()
        result = PipelineResult()
        start = time.perf_counter()

        for step in self._steps:
            step_start = time.perf_counter()
            logger.info("Pipeline step: %s", step)

            try:
                output = step._execute(context)
                context.step_outputs[step.name] = output
                result.steps_run += 1
                elapsed = time.perf_counter() - step_start
                logger.info("Step %s done in %.2fs", step.name, elapsed)
            except Exception as exc:
                elapsed = time.perf_counter() - step_start
                logger.error("Step %s failed after %.2fs: %s", step.name, elapsed, exc)
                result.errors[step.name] = str(exc)
                if stop_on_error:
                    break
                continue

        result.total_time = time.perf_counter() - start
        result.documents = context.documents
        result.step_results = context.step_outputs
        return result


# ──────────────────────────────────────────────
#  DAG Pipeline — dependency resolution + parallel
# ──────────────────────────────────────────────


class DagPipeline:
    """A DAG-based pipeline with dependency resolution, parallel branches,
    checkpointing, and resume.

    Usage::

        from runeextract.transform import DagPipeline, ExtractStep, ChunkStep, EmbedStep

        dag = DagPipeline()
        dag.add_step(ExtractStep("extract", file_path="doc.pdf"))
        dag.add_step(ChunkStep("chunk"), depends_on="extract")
        dag.add_step(EmbedStep("embed"), depends_on="chunk")

        result = dag.run()

    Args:
        steps: Initial list of (step, depends_on) pairs.
        parallel: If True, run independent branches in parallel using a
            thread pool (default False).
        max_workers: Max threads for parallel execution (default 4).
    """

    def __init__(
        self,
        *steps: Tuple[PipelineStep, Optional[Union[str, List[str]]]],
        parallel: bool = False,
        max_workers: int = 4,
    ):
        self._steps: List[PipelineStep] = []
        self._deps: Dict[str, Set[str]] = {}  # step_name -> set of dependency names
        self._completed: Set[str] = set()
        self.parallel = parallel
        self.max_workers = max_workers
        self._lock = threading.Lock()

        for step, depends_on in steps:
            self.add_step(step, depends_on=depends_on)

    def add_step(
        self,
        step: PipelineStep,
        depends_on: Optional[Union[str, List[str]]] = None,
    ) -> "DagPipeline":
        """Add a step with optional dependency declarations.

        Args:
            step: The step instance.
            depends_on: Name(s) of steps that must complete before this
                one runs. Can be a single string or list of strings.

        Returns:
            Self for chaining.
        """
        self._steps.append(step)
        if depends_on:
            if isinstance(depends_on, str):
                self._deps[step.name] = {depends_on}
            else:
                self._deps[step.name] = set(depends_on)
        else:
            self._deps[step.name] = set()
        return self

    def _topological_sort(self) -> List[PipelineStep]:
        """Kahn's algorithm — return steps in dependency-resolved order."""
        in_degree: Dict[str, int] = {}
        for s in self._steps:
            in_degree[s.name] = len(self._deps.get(s.name, set()))

        name_map = {s.name: s for s in self._steps}
        adj: Dict[str, List[str]] = {s.name: [] for s in self._steps}
        for s in self._steps:
            for dep_name in self._deps.get(s.name, set()):
                if dep_name in name_map:
                    adj[dep_name].append(s.name)

        queue = [s.name for s in self._steps if in_degree.get(s.name, 0) == 0]
        sorted_steps = []

        while queue:
            node = queue.pop(0)
            sorted_steps.append(name_map[node])
            for neighbor in adj.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_steps) != len(self._steps):
            cycle = set(s.name for s in self._steps) - set(s.name for s in sorted_steps)
            raise ValueError(f"Cycle detected in pipeline dependencies: {cycle}")

        return sorted_steps

    def _get_ready_steps(
        self, remaining: Set[str]
    ) -> List[PipelineStep]:
        """Return steps whose dependencies are all satisfied (thread-safe)."""
        with self._lock:
            ready = []
            for s in self._steps:
                if s.name not in remaining:
                    continue
                deps = self._deps.get(s.name, set())
                if deps.issubset(self._completed):
                    ready.append(s)
            return ready

    def _mark_completed(self, step_name: str) -> None:
        with self._lock:
            self._completed.add(step_name)

    def run(
        self,
        ctx: Optional[PipelineContext] = None,
        stop_on_error: bool = True,
        checkpoint_path: Optional[str] = None,
        checkpoint_interval: int = 0,
        resume_from: Optional[str] = None,
    ) -> PipelineResult:
        """Execute all steps respecting dependency order.

        Args:
            ctx: Initial context (documents, config, etc.).
            stop_on_error: If True, halt on first step failure.
            checkpoint_path: If set, save checkpoint after each step.
            checkpoint_interval: Save checkpoint every N steps (0 = every step).
            resume_from: Path to a prior checkpoint to resume from.
                Steps already recorded in the checkpoint are skipped.

        Returns:
            A ``PipelineResult`` with final documents and per-step results.
        """
        context = ctx or PipelineContext()

        if resume_from:
            if ctx is None:
                context = PipelineContext.load(resume_from)
            completed_names = set(context.step_outputs.keys())
            logger.info("Resuming from checkpoint — %d steps already done", len(completed_names))
            self._completed = completed_names

        result = PipelineResult()
        start = time.perf_counter()

        sorted_steps = self._topological_sort()
        remaining = set(s.name for s in sorted_steps)

        for s in sorted_steps:
            if s.name in self._completed:
                remaining.discard(s.name)

        if self.parallel and len(remaining) > 1:
            self._run_parallel(context, result, remaining, stop_on_error, checkpoint_path)
        else:
            for step in sorted_steps:
                if step.name not in remaining:
                    continue
                self._run_single_step(context, result, step, checkpoint_path)
                if stop_on_error and step.name in result.errors:
                    break

        result.total_time = time.perf_counter() - start
        result.documents = context.documents
        result.step_results = context.step_outputs
        return result

    def _run_single_step(
        self,
        context: PipelineContext,
        result: PipelineResult,
        step: PipelineStep,
        checkpoint_path: Optional[str],
    ) -> None:
        step_start = time.perf_counter()
        logger.info("DAG step: %s", step)

        try:
            output = step._execute(context)
            with self._lock:
                context.step_outputs[step.name] = output
                self._completed.add(step.name)
                result.steps_run += 1
            elapsed = time.perf_counter() - step_start
            logger.info("Step %s done in %.2fs", step.name, elapsed)

            if checkpoint_path:
                with self._lock:
                    context.save(checkpoint_path)
        except Exception as exc:
            elapsed = time.perf_counter() - step_start
            logger.error("Step %s failed after %.2fs: %s", step.name, elapsed, exc)
            with self._lock:
                result.errors[step.name] = str(exc)

    def _run_parallel(
        self,
        context: PipelineContext,
        result: PipelineResult,
        remaining: Set[str],
        stop_on_error: bool,
        checkpoint_path: Optional[str],
    ) -> None:
        """Execute ready steps concurrently using a thread pool."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            while True:
                with self._lock:
                    if not remaining:
                        break
                    if stop_on_error and result.errors:
                        break
                    ready = []
                    for s in self._steps:
                        if s.name not in remaining:
                            continue
                        deps = self._deps.get(s.name, set())
                        if deps.issubset(self._completed):
                            ready.append(s.name)
                            remaining.discard(s.name)
                if not ready:
                    break

                futures = {}
                name_map = {s.name: s for s in self._steps}
                for rname in ready:
                    step = name_map[rname]
                    fut = pool.submit(self._run_single_step, context, result, step, checkpoint_path)
                    futures[fut] = rname

                for future in as_completed(futures):
                    future.result()

    def _write_checkpoint(self, context: PipelineContext, path: str) -> None:
        context.save(path)

    @classmethod
    def from_pipeline(cls, pipeline: Pipeline, **kwargs) -> "DagPipeline":
        """Create a DagPipeline from a sequential Pipeline (preserves order)."""
        dag = cls(**kwargs)
        for step in pipeline._steps:
            dag.add_step(step)
        return dag


# ──────────────────────────────────────────────
#  Special step types
# ──────────────────────────────────────────────


class ConditionalStep(PipelineStep):
    """Conditionally execute one of two child steps based on a predicate.

    Args:
        name: Step name.
        condition: Callable that receives the context and returns True
            for the ``if_step`` branch.
        if_step: Step to run when condition is True.
        else_step: Optional step to run when condition is False.
    """

    def __init__(
        self,
        name: str,
        condition: Callable[[PipelineContext], bool],
        if_step: PipelineStep,
        else_step: Optional[PipelineStep] = None,
    ):
        super().__init__(name, retry_count=0, retry_delay=1.0, skip_if=None)
        self.condition = condition
        self.if_step = if_step
        self.else_step = else_step

    def run(self, ctx: PipelineContext) -> Any:
        if self.condition(ctx):
            logger.info("ConditionalStep %s → if_step %s", self.name, self.if_step.name)
            result = self.if_step._execute(ctx)
            ctx.step_outputs[self.if_step.name] = result
            return result
        elif self.else_step:
            logger.info("ConditionalStep %s → else_step %s", self.name, self.else_step.name)
            result = self.else_step._execute(ctx)
            ctx.step_outputs[self.else_step.name] = result
            return result
        else:
            logger.info("ConditionalStep %s → no branch taken", self.name)
            return None


class ParallelStep(PipelineStep):
    """Run multiple steps in parallel using a thread pool.

    All child steps share the same context.
    """

    def __init__(self, name: str, *steps: PipelineStep, max_workers: int = 4):
        super().__init__(name, retry_count=0, retry_delay=1.0, skip_if=None)
        self._children = list(steps)
        self.max_workers = max_workers

    def run(self, ctx: PipelineContext) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {}
            for child in self._children:
                fut = pool.submit(child._execute, ctx)
                futures[fut] = child.name

            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                    ctx.step_outputs[name] = results[name]
                    logger.info("ParallelStep child %s done", name)
                except Exception as exc:
                    logger.error("ParallelStep child %s failed: %s", name, exc)
                    raise
        return results


class WaitStep(PipelineStep):
    """Sleep for a specified duration (useful for rate-limiting).

    Args:
        name: Step name.
        delay: Seconds to sleep.
    """

    def __init__(self, name: str, delay: float = 1.0):
        super().__init__(name, retry_count=0, retry_delay=1.0, skip_if=None)
        self.delay = delay

    def run(self, ctx: PipelineContext) -> str:
        time.sleep(self.delay)
        msg = f"Waited {self.delay}s"
        logger.info(msg)
        return msg


# ──────────────────────────────────────────────
#  Convenience
# ──────────────────────────────────────────────


def run_pipeline(*steps: PipelineStep, **kwargs) -> PipelineResult:
    """Convenience function to create and run a pipeline in one call.

    Usage::

        from runeextract.transform import run_pipeline, ExtractStep

        result = run_pipeline(
            ExtractStep("extract", file_path="doc.pdf"),
            stop_on_error=True,
        )
    """
    pipe = Pipeline(*steps)
    return pipe.run(**kwargs)
