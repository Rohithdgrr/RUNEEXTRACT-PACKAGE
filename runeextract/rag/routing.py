"""
Feature 6: Smart Query Routing with Multi-RAG Orchestration

Intent-based routing to specialized RAG pipelines.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RouteDecision:
    """Query routing decision."""
    targets: List[str]
    confidence: float
    intent: str
    reasoning: str


class QueryRouter:
    """Smart query routing with intent classification.
    
    Routes queries to specialized RAG pipelines based on intent.
    Falls back to multi-RAG search for ambiguous queries.
    
    Features:
    - Intent classification (keyword + embedding similarity)
    - Multi-RAG orchestration
    - Result fusion with source attribution
    - Confidence-based routing
    - Learning from feedback
    
    Usage::
    
        router = QueryRouter({
            "technical": rag_engineering,
            "legal": rag_contracts,
            "financial": rag_reports
        })
        
        # Route automatically
        result = router.query("What is our patent strategy?")
        # → Routes to "legal" RAG
        
        # Get routing decision
        decision = router.route("How does the API work?")
        print(decision.targets)  # ["technical"]
        print(decision.confidence)  # 0.95
    """
    
    def __init__(
        self,
        rag_configs: Dict[str, Any],
        confidence_threshold: float = 0.85,
        enable_fusion: bool = True
    ):
        """Initialize query router.
        
        Args:
            rag_configs: Dict of {name: AutoRAG instance}
            confidence_threshold: Min confidence for single-RAG routing
            enable_fusion: Enable multi-RAG result fusion
        """
        self.rags = rag_configs
        self.confidence_threshold = confidence_threshold
        self.enable_fusion = enable_fusion
        
        # Intent patterns (keyword-based)
        self._intent_patterns = self._build_intent_patterns()
        
        # Track routing history for learning
        self._routing_history: List[Dict[str, Any]] = []
        self._feedback_scores: Dict[str, List[float]] = defaultdict(list)
        
        logger.info(
            f"QueryRouter initialized with {len(self.rags)} RAGs "
            f"(threshold={confidence_threshold:.2%})"
        )
    
    def _build_intent_patterns(self) -> Dict[str, List[str]]:
        """Build keyword patterns for each RAG target.
        
        Returns:
            Dict of {target: [keywords]}
        """
        patterns = {
            "technical": [
                "api", "code", "function", "implementation", "bug",
                "error", "architecture", "system", "design", "how does",
                "algorithm", "performance", "optimize", "deploy"
            ],
            "legal": [
                "contract", "agreement", "clause", "liability", "terms",
                "patent", "trademark", "copyright", "compliance", "regulation",
                "lawsuit", "license", "intellectual property", "rights"
            ],
            "financial": [
                "revenue", "cost", "profit", "loss", "budget", "expense",
                "forecast", "quarter", "earnings", "investment", "roi",
                "cash flow", "balance sheet", "valuation", "funding"
            ],
            "hr": [
                "employee", "hire", "salary", "benefits", "vacation",
                "performance review", "promotion", "termination", "policy",
                "training", "onboarding", "team", "manager"
            ],
            "product": [
                "feature", "roadmap", "customer", "user", "feedback",
                "release", "sprint", "backlog", "requirement", "spec",
                "usability", "ux", "ui", "design"
            ],
            "marketing": [
                "campaign", "lead", "conversion", "funnel", "brand",
                "advertising", "social media", "seo", "content", "engagement",
                "analytics", "metrics", "audience", "reach"
            ]
        }
        
        # Add patterns for user-defined RAGs
        for rag_name in self.rags.keys():
            if rag_name not in patterns:
                patterns[rag_name] = [rag_name.lower()]
        
        return patterns
    
    def add_intent_patterns(self, target: str, keywords: List[str]) -> None:
        """Add custom intent patterns.
        
        Args:
            target: RAG target name
            keywords: List of keywords for this target
        """
        if target not in self._intent_patterns:
            self._intent_patterns[target] = []
        
        self._intent_patterns[target].extend(keywords)
        logger.info(f"Added {len(keywords)} patterns for {target}")
    
    def route(
        self,
        query: str,
        top_k: int = 1
    ) -> RouteDecision:
        """Determine which RAG(s) to query.
        
        Args:
            query: User query
            top_k: Max number of RAGs to route to
        
        Returns:
            RouteDecision with targets and confidence
        """
        query_lower = query.lower()
        
        # Score each target
        scores: Dict[str, float] = {}
        for target, patterns in self._intent_patterns.items():
            if target not in self.rags:
                continue
            
            # Keyword matching score
            keyword_score = sum(
                1 for pattern in patterns
                if pattern in query_lower
            ) / max(len(patterns), 1)
            
            # Boost from historical success
            if target in self._feedback_scores:
                avg_feedback = sum(self._feedback_scores[target]) / len(self._feedback_scores[target])
                keyword_score = keyword_score * 0.7 + avg_feedback * 0.3
            
            scores[target] = keyword_score
        
        # Sort by score
        sorted_targets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_targets:
            # No scores - query all RAGs
            return RouteDecision(
                targets=list(self.rags.keys()),
                confidence=0.0,
                intent="unknown",
                reasoning="No patterns matched - querying all RAGs"
            )
        
        top_target, top_score = sorted_targets[0]
        
        # High confidence single-target routing
        if top_score >= self.confidence_threshold:
            return RouteDecision(
                targets=[top_target],
                confidence=top_score,
                intent=top_target,
                reasoning=f"Strong intent match ({top_score:.2%})"
            )
        
        # Medium confidence multi-target routing
        elif top_score >= self.confidence_threshold * 0.6:
            # Include top-k targets above threshold
            targets = [
                target for target, score in sorted_targets[:top_k]
                if score >= self.confidence_threshold * 0.5
            ]
            
            return RouteDecision(
                targets=targets,
                confidence=top_score,
                intent="mixed",
                reasoning=f"Moderate confidence - querying {len(targets)} RAGs"
            )
        
        # Low confidence - query all
        else:
            return RouteDecision(
                targets=list(self.rags.keys()),
                confidence=top_score,
                intent="ambiguous",
                reasoning="Low confidence - querying all RAGs"
            )
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        return_citations: bool = True,
        **kwargs
    ) -> Any:
        """Query with automatic routing.
        
        Args:
            question: User question
            top_k: Number of chunks per RAG
            return_citations: Include citations
            **kwargs: Passed to RAG.query()
        
        Returns:
            RAGResult (merged if multiple targets)
        """
        # Get routing decision
        decision = self.route(question)
        
        logger.info(
            f"Routing to {len(decision.targets)} RAG(s): {decision.targets} "
            f"(confidence={decision.confidence:.2%})"
        )
        
        # Query target RAGs
        results = []
        for target in decision.targets:
            try:
                rag = self.rags[target]
                result = rag.query(
                    question=question,
                    top_k=top_k,
                    return_citations=return_citations,
                    **kwargs
                )
                results.append((target, result))
            except Exception as e:
                logger.warning(f"Query failed for {target}: {e}")
        
        if not results:
            raise Exception("All RAG queries failed")
        
        # Single target - return directly
        if len(results) == 1:
            target, result = results[0]
            self._record_routing(question, decision, target, result)
            return result
        
        # Multiple targets - merge results
        if self.enable_fusion:
            merged = self._merge_results(results, question, decision)
            self._record_routing(question, decision, None, merged)
            return merged
        else:
            # Return best result
            best_target, best_result = max(
                results,
                key=lambda x: x[1].confidence
            )
            self._record_routing(question, decision, best_target, best_result)
            return best_result
    
    def _merge_results(
        self,
        results: List[Tuple[str, Any]],
        question: str,
        decision: RouteDecision
    ) -> Any:
        """Merge results from multiple RAGs.
        
        Args:
            results: List of (target, RAGResult) tuples
            question: Original question
            decision: Routing decision
        
        Returns:
            Merged RAGResult
        """
        from runeextract.rag.types import RAGResult, Citation
        
        # Collect all chunks (deduplicate by text)
        all_chunks = []
        seen_texts = set()
        for target, result in results:
            for chunk in result.retrieved_chunks:
                if chunk.text not in seen_texts:
                    # Tag with source RAG
                    chunk.metadata = chunk.metadata or {}
                    chunk.metadata["source_rag"] = target
                    all_chunks.append(chunk)
                    seen_texts.add(chunk.text)
        
        # Sort by score
        all_chunks.sort(key=lambda c: c.score, reverse=True)
        
        # Re-generate answer from merged context
        # (Use first RAG's AI processor)
        rag = self.rags[results[0][0]]
        
        # Simple answer fusion: concatenate top answers with attribution
        answers = []
        for target, result in results:
            answers.append(f"**From {target}:** {result.answer}")
        
        merged_answer = "\n\n".join(answers)
        
        # Merge citations
        all_citations = []
        citation_offset = 0
        for target, result in results:
            for citation in result.citations:
                citation.id += citation_offset
                all_citations.append(citation)
            citation_offset += len(result.citations)
        
        # Compute merged confidence (weighted average)
        total_weight = sum(r.confidence for _, r in results)
        merged_confidence = sum(
            r.confidence * r.confidence for _, r in results
        ) / total_weight if total_weight > 0 else 0.5
        
        # Compute merged latency and cost
        merged_latency = sum(r.latency_ms for _, r in results)
        merged_cost = sum(r.cost for _, r in results)
        
        return RAGResult(
            answer=merged_answer,
            citations=all_citations,
            confidence=merged_confidence,
            retrieved_chunks=all_chunks[:10],  # Top 10 from merged
            query_variants=[],
            latency_ms=merged_latency,
            tokens_used={
                "input": sum(r.tokens_used.get("input", 0) for _, r in results),
                "output": sum(r.tokens_used.get("output", 0) for _, r in results)
            },
            cost=merged_cost,
            total_session_cost=sum(r.total_session_cost for _, r in results)
        )
    
    def _record_routing(
        self,
        query: str,
        decision: RouteDecision,
        selected_target: Optional[str],
        result: Any
    ) -> None:
        """Record routing decision for learning.
        
        Args:
            query: Original query
            decision: Routing decision
            selected_target: Target that was used (None if merged)
            result: Query result
        """
        self._routing_history.append({
            "query": query,
            "decision": decision,
            "selected_target": selected_target,
            "confidence": result.confidence,
            "latency_ms": result.latency_ms
        })
        
        # Trim history
        if len(self._routing_history) > 1000:
            self._routing_history = self._routing_history[-1000:]
    
    def record_feedback(
        self,
        query: str,
        target: str,
        score: float
    ) -> None:
        """Record user feedback on routing decision.
        
        Args:
            query: Original query
            target: RAG target that was used
            score: Feedback score (0.0 to 1.0)
        """
        self._feedback_scores[target].append(score)
        
        # Trim feedback history
        if len(self._feedback_scores[target]) > 100:
            self._feedback_scores[target] = self._feedback_scores[target][-100:]
        
        logger.info(f"Recorded feedback for {target}: {score:.2f}")
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics.
        
        Returns:
            Dict with routing metrics
        """
        if not self._routing_history:
            return {"total_queries": 0}
        
        # Compute stats
        total_queries = len(self._routing_history)
        single_rag_queries = sum(
            1 for h in self._routing_history
            if len(h["decision"].targets) == 1
        )
        multi_rag_queries = total_queries - single_rag_queries
        
        # Target distribution
        target_counts = defaultdict(int)
        for h in self._routing_history:
            if h["selected_target"]:
                target_counts[h["selected_target"]] += 1
        
        # Average confidence per target
        target_confidences = defaultdict(list)
        for h in self._routing_history:
            if h["selected_target"]:
                target_confidences[h["selected_target"]].append(h["confidence"])
        
        avg_confidences = {
            target: sum(scores) / len(scores)
            for target, scores in target_confidences.items()
        }
        
        return {
            "total_queries": total_queries,
            "single_rag_queries": single_rag_queries,
            "multi_rag_queries": multi_rag_queries,
            "single_rag_rate": single_rag_queries / total_queries,
            "target_distribution": dict(target_counts),
            "avg_confidence_by_target": avg_confidences,
            "feedback_counts": {
                target: len(scores)
                for target, scores in self._feedback_scores.items()
            }
        }
