"""Pipeline DAG — define and execute document processing workflows."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from runeextract.models.document import Document

logger = logging.getLogger(__name__)


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


@dataclass
class PipelineResult:
    """Result of a completed pipeline run."""
    documents: List[Document] = field(default_factory=list)
    step_results: Dict[str, Any] = field(default_factory=dict)
    total_time: float = 0.0
    steps_run: int = 0
    errors: Dict[str, str] = field(default_factory=dict)


class PipelineStep:
    """Base class for a single pipeline processing step.

    Subclasses must implement ``run(self, ctx: PipelineContext)``.
    """

    def __init__(self, name: str, **kwargs):
        self.name = name
        self._kwargs = kwargs

    def run(self, ctx: PipelineContext) -> Any:
        """Execute this step.

        Args:
            ctx: Current pipeline context with documents and previous outputs.

        Returns:
            Arbitrary data stored in ``context.step_outputs[self.name]``.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


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

    def __init__(self, *steps: PipelineStep):
        if not steps:
            raise ValueError("Pipeline requires at least one step")
        self._steps = list(steps)

    def add_step(self, step: PipelineStep, index: Optional[int] = None) -> "Pipeline":
        """Add a step at an optional index (default: end)."""
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
        """Execute all pipeline steps in order.

        Args:
            ctx: Initial context (documents, config, etc.).
            stop_on_error: If True, halt on first step failure.

        Returns:
            A ``PipelineResult`` with final documents and per-step results.
        """
        context = ctx or PipelineContext()
        result = PipelineResult()
        start = time.perf_counter()

        for step in self._steps:
            step_start = time.perf_counter()
            logger.info("Pipeline step: %s", step)

            try:
                output = step.run(context)
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
