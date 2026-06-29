"""
QueryAnalyzer — intelligently determine retrieval strategy from query features.

Analyzes query length, question type, complexity, and vocabulary density
to recommend whether HyDE, MultiQuery, or a specific top_k should be used.
"""

import re
from typing import Dict, Any


class QueryAnalyzer:
    """Analyze a query to recommend optimal retrieval settings.

    The analyzer examines the query's lexical and structural characteristics
    to determine:
    - Question type (factual, analytical, comparative, exploratory)
    - Whether HyDE would help (hypothetical document embedding)
    - Whether MultiQuery would help (query expansion)
    - Optimal top_k (broader retrieval for complex queries)

    Usage::

        analyzer = QueryAnalyzer()
        result = analyzer.analyze("Why did the experiment fail?")
        # result["recommend_hyde"] -> True
        # result["recommend_multi_query"] -> True
        # result["question_type"] -> "analytical"
    """

    _FACTUAL_INDICATORS = {"what", "when", "where", "who", "which", "how many", "how much", "how long", "define", "list", "name"}
    _ANALYTICAL_INDICATORS = {"why", "how does", "explain", "analyze", "what causes", "what is the reason", "what are the factors"}
    _COMPARATIVE_INDICATORS = {"compare", "difference", "versus", "vs", "better", "worse", "similarities", "differences", "pros", "cons"}
    _EXPLORATORY_INDICATORS = {"summarize", "overview", "tell me about", "describe", "what is", "what are", "outline"}

    def analyze(self, question: str) -> Dict[str, Any]:
        """Analyze a query and return recommended retrieval settings.

        Returns:
            Dict with keys: ``length``, ``complexity``, ``question_type``,
            ``recommend_hyde``, ``recommend_multi_query``, ``recommend_top_k``,
            ``lexical_density``, ``term_count``.
        """
        q_lower = question.lower().strip()
        tokens = re.findall(r"\w+", q_lower)
        term_count = len(tokens)
        unique_terms = len(set(tokens))
        lexical_density = unique_terms / max(term_count, 1)

        question_type = self._detect_type(q_lower)
        complexity = self._compute_complexity(question, term_count, unique_terms, question_type)

        recommend_hyde = self._should_hyde(question_type, complexity, term_count)
        recommend_multi_query = self._should_multi_query(question_type, complexity, term_count)
        recommend_top_k = self._compute_top_k(question_type, complexity)

        return {
            "length": term_count,
            "complexity": complexity,
            "question_type": question_type,
            "recommend_hyde": recommend_hyde,
            "recommend_multi_query": recommend_multi_query,
            "recommend_top_k": recommend_top_k,
            "lexical_density": lexical_density,
            "term_count": term_count,
        }

    def _detect_type(self, q_lower: str) -> str:
        for word in self._COMPARATIVE_INDICATORS:
            if word in q_lower:
                return "comparative"
        for word in self._ANALYTICAL_INDICATORS:
            if word in q_lower:
                return "analytical"
        for word in self._FACTUAL_INDICATORS:
            if word in q_lower:
                return "factual"
        for word in self._EXPLORATORY_INDICATORS:
            if word in q_lower:
                return "exploratory"
        if len(q_lower.split()) <= 2:
            return "keyword"
        return "exploratory"

    def _compute_complexity(self, question: str, term_count: int, unique_terms: int, qtype: str) -> float:
        score = 0.0
        if term_count > 10:
            score += 0.3
        if unique_terms > 8:
            score += 0.2
        if self._has_qualifiers(question):
            score += 0.2
        if "?" in question:
            score += 0.1
        if qtype in ("analytical", "comparative"):
            score += 0.2
        return min(1.0, score)

    @staticmethod
    def _has_qualifiers(text: str) -> bool:
        qualifiers = {"specifically", "especially", "primarily", "particularly",
                       "mostly", "mainly", "including", "such as", "in terms of"}
        return any(q in text.lower() for q in qualifiers)

    @staticmethod
    def _should_hyde(qtype: str, complexity: float, term_count: int) -> bool:
        if qtype in ("analytical", "comparative"):
            return True
        if complexity > 0.5:
            return True
        if term_count <= 3 and qtype == "keyword":
            return True
        return False

    @staticmethod
    def _should_multi_query(qtype: str, complexity: float, term_count: int) -> bool:
        if qtype in ("analytical", "comparative"):
            return True
        if complexity > 0.6:
            return True
        if qtype == "exploratory" and term_count > 5:
            return True
        return False

    @staticmethod
    def _compute_top_k(qtype: str, complexity: float) -> int:
        if qtype == "comparative":
            return 10
        if qtype == "analytical":
            return 7
        if complexity > 0.6:
            return 7
        if qtype == "keyword":
            return 3
        return 5
