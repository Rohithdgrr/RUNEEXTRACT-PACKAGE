"""Export adapters for RAG evaluation and framework formats."""

import enum
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExportError(Exception):
    """Raised when export fails."""


class ExportFormat(str, enum.Enum):
    RAGAS = "ragas"
    DSPY = "dspy"
    HAYSTACK = "haystack"
    LLAMA_INDEX = "llama_index"
    JSONL = "jsonl"


@dataclass
class ExportRecord:
    question: str = ""
    answer: str = ""
    contexts: List[str] = field(default_factory=list)
    ground_truth: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    document_id: str = ""


def to_ragas(records: List[ExportRecord], output_path: Optional[str] = None) -> str:
    """Export to RAGAS JSON format (qa.json / ground_truths.json)."""
    result = {
        "questions": [r.question for r in records],
        "answers": [r.answer for r in records],
        "contexts": [r.contexts for r in records],
        "ground_truths": [[r.ground_truth] if r.ground_truth else [] for r in records],
    }
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return json.dumps(result, ensure_ascii=False, indent=2)


def to_dspy(records: List[ExportRecord], output_path: Optional[str] = None) -> str:
    """Export to DSPy-compatible JSON format."""
    result = []
    for r in records:
        entry = {"question": r.question, "answer": r.answer}
        if r.contexts:
            entry["context"] = "\n\n".join(r.contexts)
        if r.ground_truth:
            entry["ground_truth"] = r.ground_truth
        if r.metadata:
            entry["metadata"] = r.metadata
        result.append(entry)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return json.dumps(result, ensure_ascii=False, indent=2)


def to_haystack(records: List[ExportRecord], output_path: Optional[str] = None) -> str:
    """Export to Haystack Document format."""
    result = {"documents": []}
    for i, r in enumerate(records):
        doc = {
            "id": r.document_id or f"doc_{i}",
            "content": r.answer or r.question,
            "content_type": "text",
            "meta": {
                "question": r.question,
                "source": r.source,
                **(r.metadata or {}),
            },
        }
        if r.contexts:
            doc["meta"]["contexts"] = r.contexts
        if r.ground_truth:
            doc["meta"]["ground_truth"] = r.ground_truth
        result["documents"].append(doc)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return json.dumps(result, ensure_ascii=False, indent=2)


def to_llama_index(records: List[ExportRecord], output_path: Optional[str] = None) -> str:
    """Export to LlamaIndex-compatible JSON format."""
    result = []
    for i, r in enumerate(records):
        entry = {
            "id_": r.document_id or f"doc_{i}",
            "text": r.answer or r.question,
            "metadata": {
                "question": r.question,
                "source": r.source,
                **(r.metadata or {}),
            },
        }
        if r.contexts:
            entry["metadata"]["contexts"] = r.contexts
        if r.ground_truth:
            entry["metadata"]["ground_truth"] = r.ground_truth
        result.append(entry)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return json.dumps(result, ensure_ascii=False, indent=2)


def to_jsonl(records: List[ExportRecord], output_path: str) -> str:
    """Export to JSONL format (one record per line)."""
    lines = []
    for r in records:
        lines.append(json.dumps(asdict(r), ensure_ascii=False))
    result = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)
    if not result.endswith("\n"):
        result += "\n"
    return result


_FORMAT_MAP = {
    ExportFormat.RAGAS: to_ragas,
    ExportFormat.DSPY: to_dspy,
    ExportFormat.HAYSTACK: to_haystack,
    ExportFormat.LLAMA_INDEX: to_llama_index,
    ExportFormat.JSONL: to_jsonl,
}


def export_document(
    records: List[ExportRecord],
    fmt: ExportFormat,
    output_path: Optional[str] = None,
) -> str:
    exporter = _FORMAT_MAP.get(fmt)
    if exporter is None:
        raise ExportError(f"Unsupported export format: {fmt}")
    return exporter(records, output_path=output_path)
