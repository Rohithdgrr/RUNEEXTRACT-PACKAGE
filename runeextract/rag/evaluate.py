"""
RAG pipeline evaluation: auto-generate test sets and compute quality metrics.
"""

import logging
import random
from typing import List, Dict, Any, Optional, Callable, Tuple

from runeextract.models.document import Document
from runeextract.utils.maturity import beta

logger = logging.getLogger(__name__)


@beta(name="rag.evaluate")
class RAGEvaluator:
    """Evaluate RAG pipeline quality with auto-generated test sets."""

    def __init__(self, query_fn: Optional[Callable] = None,
                 llm_complete: Optional[Callable] = None):
        self.query_fn = query_fn
        self.llm_complete = llm_complete

    def generate_test_set(self, documents: List[Document],
                          num_questions: int = 50,
                          seed: int = 42) -> List[Dict[str, str]]:
        """Auto-generate Q&A pairs from document chunks.

        Samples chunks across all documents and uses an LLM to produce
        questions that can be answered solely from each chunk.

        Returns:
            List of dicts with keys: question, answer, chunk_text, source
        """
        if not self.llm_complete:
            logger.warning("No LLM available for test set generation")
            return []

        all_chunks = []
        for doc in documents:
            chunks = doc.chunks(strategy="fixed_size", size=500)
            for c in chunks:
                all_chunks.append((c, doc))

        rng = random.Random(seed)
        sample = rng.sample(all_chunks, min(num_questions, len(all_chunks)))
        test_cases = []

        for chunk, doc in sample:
            prompt = (
                "Generate one question that can be answered SOLELY from the text below. "
                "Include the exact answer as it appears in the text.\n\n"
                f"Text:\n{chunk.text[:1500]}\n\n"
                "Format:\nQ: <question>\nA: <exact answer from text>"
            )
            try:
                response = self.llm_complete(prompt, max_tokens=300)
                q, a = self._parse_qa(response)
                if q and a:
                    test_cases.append({
                        "question": q,
                        "answer": a,
                        "chunk_text": chunk.text,
                        "source": doc.source_path or "",
                    })
            except Exception as e:
                logger.debug(f"Test generation failed: {e}")

        return test_cases

    def evaluate(self, test_set: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
        """Run evaluation and return per-metric aggregates.

        Metrics:
            - answer_relevance: lexical overlap between answer and question
            - answer_relevance_llm: LLM-judged relevance (when ``llm_complete`` set)
            - context_precision: proportion of retrieved chunks containing the answer
            - faithfulness: lexical overlap between answer and context
            - faithfulness_llm: LLM-judged faithfulness (when ``llm_complete`` set)
            - answer_similarity: Jaccard similarity between generated and expected answer

        Requires ``self.query_fn`` to be set.
        """
        if not self.query_fn:
            return {"error": {"mean": 0.0, "message": "query_fn not set"}}

        raw: Dict[str, List[float]] = {
            "answer_relevance": [],
            "answer_relevance_llm": [],
            "context_precision": [],
            "faithfulness": [],
            "faithfulness_llm": [],
            "answer_similarity": [],
        }

        for tc in test_set:
            try:
                result = self.query_fn(tc["question"], top_k=5, return_citations=True)
            except Exception as exc:
                logger.warning("Evaluation query failed: %s", exc)
                continue
            retrieved = [c.text for c in result.retrieved_chunks]
            context = "\n".join(retrieved)

            raw["answer_relevance"].append(
                self._rate_relevance(result.answer, tc["question"])
            )
            raw["answer_relevance_llm"].append(
                self._rate_relevance_llm(result.answer, tc["question"])
            )
            raw["context_precision"].append(
                self._has_answer(retrieved, tc["answer"])
            )
            raw["faithfulness"].append(
                self._rate_faithfulness(result.answer, context)
            )
            raw["faithfulness_llm"].append(
                self._rate_faithfulness_llm(result.answer, context)
            )
            raw["answer_similarity"].append(
                self._semantic_similarity(result.answer, tc["answer"])
            )

        return {
            metric: self._aggregate(scores)
            for metric, scores in raw.items()
            if scores
        }

    def _parse_qa(self, response: str) -> Tuple[Optional[str], Optional[str]]:
        lines = response.strip().split("\n")
        q = None
        a = None
        for line in lines:
            if line.startswith("Q:") or line.startswith("Q "):
                q = line.split(":", 1)[1].strip() if ":" in line else None
            if line.startswith("A:") or line.startswith("A "):
                a = line.split(":", 1)[1].strip() if ":" in line else None
        return q, a

    def _rate_relevance_llm(self, answer: str, question: str) -> float:
        """LLM-judged relevance when an LLM is available."""
        if not self.llm_complete:
            return 0.0
        try:
            prompt = (
                "Rate the relevance of the ANSWER to the QUESTION "
                "on a scale of 0.0 to 1.0. Only respond with the number.\n\n"
                f"QUESTION: {question}\nANSWER: {answer}"
            )
            response = self.llm_complete(prompt, max_tokens=10)
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.0

    def _rate_faithfulness_llm(self, answer: str, context: str) -> float:
        """LLM-judged faithfulness — how much of the answer is supported."""
        if not self.llm_complete or not answer or not context:
            return 0.0 if not answer else self._rate_faithfulness(answer, context)
        try:
            prompt = (
                "Rate how much of the ANSWER is supported by the CONTEXT "
                "on a scale of 0.0 to 1.0. Only respond with the number.\n\n"
                f"CONTEXT: {context[:2000]}\nANSWER: {answer}"
            )
            response = self.llm_complete(prompt, max_tokens=10)
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except Exception:
            return self._rate_faithfulness(answer, context)

    def _rate_relevance(self, answer: str, question: str) -> float:
        """Simple lexical relevance: word overlap ratio."""
        a_words = set(answer.lower().split())
        q_words = set(question.lower().split())
        if not a_words or not q_words:
            return 0.0
        overlap = a_words & q_words
        return len(overlap) / len(q_words)

    def _has_answer(self, retrieved: List[str], expected: str) -> float:
        """Fraction of retrieved chunks that contain the expected answer."""
        if not retrieved or not expected:
            return 0.0
        keywords = set(expected.lower().split())
        found = sum(1 for t in retrieved if any(kw in t.lower() for kw in keywords))
        return found / len(retrieved)

    def _rate_faithfulness(self, answer: str, context: str) -> float:
        """Score how much of the answer is supported by context (word overlap)."""
        if not answer or not context:
            return 0.0
        a_words = set(answer.lower().split())
        c_words = set(context.lower().split())
        if not a_words:
            return 0.0
        supported = sum(1 for w in a_words if w in c_words)
        return supported / len(a_words)

    def _semantic_similarity(self, a: str, b: str) -> float:
        """Simple Jaccard similarity on word sets."""
        if not a or not b:
            return 0.0
        a_set = set(a.lower().split())
        b_set = set(b.lower().split())
        if not a_set or not b_set:
            return 0.0
        return len(a_set & b_set) / len(a_set | b_set)

    @staticmethod
    def _aggregate(scores: List[float]) -> Dict[str, float]:
        if not scores:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        import numpy as np
        arr = np.array(scores)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": len(scores),
        }
