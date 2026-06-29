"""RAG Evaluation Suite — LLM-judged relevance, retrieval accuracy, scorecard."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EvalQuestion:
    question: str
    ground_truth: str = ""
    expected_source: str = ""


@dataclass
class EvalResult:
    question: str
    answer: str = ""
    retrieved_chunks: List[str] = field(default_factory=list)
    relevance_score: float = 0.0
    retrieval_accuracy: float = 0.0
    latency_ms: float = 0.0
    answer_coverage: float = 0.0


@dataclass
class Scorecard:
    total_questions: int = 0
    avg_relevance: float = 0.0
    avg_retrieval_accuracy: float = 0.0
    avg_latency_ms: float = 0.0
    avg_answer_coverage: float = 0.0
    results: List[EvalResult] = field(default_factory=list)

    def print(self):
        print(f"Total questions: {self.total_questions}")
        print(f"Average relevance: {self.avg_relevance:.2f}")
        print(f"Average retrieval accuracy: {self.avg_retrieval_accuracy:.2f}")
        print(f"Average latency: {self.avg_latency_ms:.0f}ms")
        print(f"Average answer coverage: {self.avg_answer_coverage:.2f}")


class RAGEvaluator:
    def __init__(self, rag: Any, judge_llm: Optional[Any] = None):
        self._rag = rag
        self._judge = judge_llm

    def _judge_relevance(self, question: str, answer: str) -> float:
        if self._judge:
            try:
                resp = self._judge(
                    "Rate answer relevance from 0 to 1. Return only the number.",
                    f"Question: {question}\nAnswer: {answer}",
                )
                return max(0.0, min(1.0, float(resp.strip())))
            except Exception:
                pass
        return 0.5

    def _judge_coverage(self, question: str, answer: str, ground_truth: str) -> float:
        if not ground_truth:
            return 0.0
        overlap = len(set(answer.lower().split()) & set(ground_truth.lower().split()))
        total = len(set(ground_truth.lower().split())) or 1
        return min(1.0, overlap / total)

    def evaluate_question(self, q: EvalQuestion, top_k: int = 5) -> EvalResult:
        start = time.time()
        result = self._rag.query(q.question)
        elapsed = time.time() - start
        answer = getattr(result, "answer", "") or str(result)
        chunks = [c.text for c in getattr(result, "retrieved_chunks", [])] if hasattr(result, "retrieved_chunks") else []
        relevance = self._judge_relevance(q.question, answer)
        coverage = self._judge_coverage(q.question, answer, q.ground_truth) if q.ground_truth else 0.0
        retrieval_acc = 1.0 if (not q.expected_source or any(q.expected_source in c for c in chunks)) else 0.0
        return EvalResult(
            question=q.question,
            answer=answer,
            retrieved_chunks=chunks,
            relevance_score=relevance,
            retrieval_accuracy=retrieval_acc,
            latency_ms=elapsed * 1000,
            answer_coverage=coverage,
        )

    def run(self, questions: List[EvalQuestion], top_k: int = 5) -> Scorecard:
        results = [self.evaluate_question(q, top_k=top_k) for q in questions]
        n = len(results) or 1
        return Scorecard(
            total_questions=len(results),
            avg_relevance=sum(r.relevance_score for r in results) / n,
            avg_retrieval_accuracy=sum(r.retrieval_accuracy for r in results) / n,
            avg_latency_ms=sum(r.latency_ms for r in results) / n,
            avg_answer_coverage=sum(r.answer_coverage for r in results) / n,
            results=results,
        )
