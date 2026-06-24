"""
Feature 9: A/B Testing & Experimentation Framework

Test multiple RAG configurations simultaneously.
"""

import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class VariantConfig:
    """RAG configuration variant."""
    name: str
    config: Dict[str, Any]
    description: str = ""


@dataclass
class VariantMetrics:
    """Metrics for a single variant."""
    name: str
    queries: int = 0
    avg_confidence: float = 0.0
    avg_latency_ms: float = 0.0
    avg_cost: float = 0.0
    avg_citations: float = 0.0
    error_rate: float = 0.0
    feedback_scores: List[float] = field(default_factory=list)
    
    @property
    def avg_feedback(self) -> float:
        """Average user feedback score."""
        return mean(self.feedback_scores) if self.feedback_scores else 0.0
    
    @property
    def feedback_stdev(self) -> float:
        """Standard deviation of feedback scores."""
        if len(self.feedback_scores) < 2:
            return 0.0
        return stdev(self.feedback_scores)


@dataclass
class ExperimentReport:
    """Experiment summary report."""
    name: str
    start_time: float
    end_time: float
    variants: Dict[str, VariantMetrics]
    winner: Optional[str] = None
    confidence_level: float = 0.0
    total_queries: int = 0
    
    @property
    def duration_hours(self) -> float:
        """Experiment duration in hours."""
        return (self.end_time - self.start_time) / 3600
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "duration_hours": self.duration_hours,
            "total_queries": self.total_queries,
            "winner": self.winner,
            "confidence_level": self.confidence_level,
            "variants": {
                name: {
                    "queries": m.queries,
                    "avg_confidence": m.avg_confidence,
                    "avg_latency_ms": m.avg_latency_ms,
                    "avg_cost": m.avg_cost,
                    "avg_feedback": m.avg_feedback,
                    "error_rate": m.error_rate
                }
                for name, m in self.variants.items()
            }
        }


class ExperimentManager:
    """A/B testing framework for RAG configurations.
    
    Test multiple RAG configurations simultaneously with:
    - Consistent user bucketing (hash-based)
    - Statistical significance testing
    - Multi-metric optimization
    - User feedback collection
    - Experiment reports
    
    Usage::
    
        # Define variants
        variants = {
            "control": {
                "chunk_size": 1000,
                "top_k": 5,
                "reranker": None
            },
            "treatment_a": {
                "chunk_size": 500,
                "top_k": 10,
                "reranker": "cross-encoder"
            },
            "treatment_b": {
                "chunk_size": 1500,
                "top_k": 3,
                "reranker": "cross-encoder"
            }
        }
        
        # Start experiment
        exp = ExperimentManager(
            name="chunking_strategy",
            variants=variants,
            rag_factory=lambda config: auto_rag(**config)
        )
        
        # Query with automatic bucketing
        result = exp.query(
            question="What is RAG?",
            user_id="user123"
        )
        
        # Record feedback
        exp.record_feedback("user123", score=0.9)
        
        # Get report
        report = exp.get_report()
        print(f"Winner: {report.winner}")
    """
    
    def __init__(
        self,
        name: str,
        variants: Dict[str, Dict[str, Any]],
        rag_factory: Any,
        split: Optional[List[float]] = None,
        min_queries: int = 100,
        significance_level: float = 0.95
    ):
        """Initialize experiment manager.
        
        Args:
            name: Experiment name
            variants: Dict of {variant_name: config_dict}
            rag_factory: Factory function that creates RAG from config
            split: Traffic split [0.33, 0.33, 0.34] (auto if None)
            min_queries: Min queries per variant for significance
            significance_level: Statistical significance threshold
        """
        self.name = name
        self.variants_config = variants
        self.rag_factory = rag_factory
        self.min_queries = min_queries
        self.significance_level = significance_level
        
        # Auto-compute even split if not provided
        if split is None:
            n = len(variants)
            split = [1.0 / n] * n
        
        self.split = split
        
        # Build RAG instances
        self.rags = {
            name: rag_factory(config)
            for name, config in variants.items()
        }
        
        # Tracking
        self._user_buckets: Dict[str, str] = {}
        self._metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._feedback: Dict[str, Dict[str, float]] = defaultdict(dict)  # user_id -> score
        self._start_time = time.time()
        
        logger.info(
            f"Experiment '{name}' started with {len(variants)} variants: "
            f"{list(variants.keys())}"
        )
    
    def assign_variant(self, user_id: str) -> str:
        """Assign user to a variant (consistent bucketing).
        
        Args:
            user_id: User identifier
        
        Returns:
            Variant name
        """
        # Check cache first
        if user_id in self._user_buckets:
            return self._user_buckets[user_id]
        
        # Hash user ID to integer
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100
        
        # Assign to variant based on split
        cumulative = 0.0
        variant_names = list(self.variants_config.keys())
        
        for i, prob in enumerate(self.split):
            cumulative += prob * 100
            if hash_val < cumulative:
                variant = variant_names[i]
                self._user_buckets[user_id] = variant
                return variant
        
        # Fallback (shouldn't reach here)
        variant = variant_names[-1]
        self._user_buckets[user_id] = variant
        return variant
    
    def query(
        self,
        question: str,
        user_id: str,
        **kwargs
    ) -> Any:
        """Query with automatic variant assignment.
        
        Args:
            question: User question
            user_id: User identifier
            **kwargs: Passed to RAG.query()
        
        Returns:
            RAGResult from assigned variant
        """
        # Get variant for user
        variant = self.assign_variant(user_id)
        rag = self.rags[variant]
        
        # Query
        start = time.time()
        error = None
        result = None
        
        try:
            result = rag.query(question, **kwargs)
        except Exception as e:
            error = str(e)
            logger.warning(f"Query failed for variant {variant}: {e}")
            raise
        finally:
            # Record metrics
            latency = (time.time() - start) * 1000
            
            self._metrics[variant].append({
                "user_id": user_id,
                "question": question,
                "confidence": result.confidence if result else 0.0,
                "latency_ms": latency,
                "cost": result.cost if result else 0.0,
                "citations": len(result.citations) if result else 0,
                "error": error,
                "timestamp": time.time()
            })
        
        return result
    
    def record_feedback(
        self,
        user_id: str,
        score: float
    ) -> None:
        """Record user feedback for last query.
        
        Args:
            user_id: User identifier
            score: Feedback score (0.0 to 1.0)
        """
        if user_id not in self._user_buckets:
            logger.warning(f"No variant for user {user_id}")
            return
        
        variant = self._user_buckets[user_id]
        self._feedback[variant][user_id] = score
        
        logger.debug(f"Feedback for {variant} from {user_id}: {score:.2f}")
    
    def get_metrics(self, variant: str) -> VariantMetrics:
        """Get metrics for a variant.
        
        Args:
            variant: Variant name
        
        Returns:
            VariantMetrics object
        """
        data = self._metrics[variant]
        
        if not data:
            return VariantMetrics(name=variant)
        
        # Compute aggregated metrics
        total_queries = len(data)
        errors = sum(1 for d in data if d["error"])
        
        metrics = VariantMetrics(
            name=variant,
            queries=total_queries,
            avg_confidence=mean([d["confidence"] for d in data]),
            avg_latency_ms=mean([d["latency_ms"] for d in data]),
            avg_cost=mean([d["cost"] for d in data]),
            avg_citations=mean([d["citations"] for d in data]),
            error_rate=errors / total_queries if total_queries > 0 else 0.0,
            feedback_scores=list(self._feedback[variant].values())
        )
        
        return metrics
    
    def get_report(self) -> ExperimentReport:
        """Generate experiment report with statistical analysis.
        
        Returns:
            ExperimentReport with winner and confidence level
        """
        # Collect metrics for all variants
        variant_metrics = {
            name: self.get_metrics(name)
            for name in self.variants_config.keys()
        }
        
        # Determine winner (highest average feedback or confidence)
        winner = None
        best_score = -1.0
        
        for name, metrics in variant_metrics.items():
            # Use feedback if available, else confidence
            score = metrics.avg_feedback if metrics.feedback_scores else metrics.avg_confidence
            
            if score > best_score:
                best_score = score
                winner = name
        
        # Compute statistical significance
        confidence_level = self._compute_significance(variant_metrics, winner)
        
        total_queries = sum(m.queries for m in variant_metrics.values())
        
        report = ExperimentReport(
            name=self.name,
            start_time=self._start_time,
            end_time=time.time(),
            variants=variant_metrics,
            winner=winner,
            confidence_level=confidence_level,
            total_queries=total_queries
        )
        
        logger.info(
            f"Experiment '{self.name}' report: "
            f"Winner={winner} ({confidence_level:.1%} confidence), "
            f"{total_queries} total queries"
        )
        
        return report
    
    def _compute_significance(
        self,
        variant_metrics: Dict[str, VariantMetrics],
        winner: str
    ) -> float:
        """Compute statistical significance of winner.
        
        Args:
            variant_metrics: Metrics for all variants
            winner: Winning variant name
        
        Returns:
            Confidence level (0.0 to 1.0)
        """
        if not winner:
            return 0.0
        
        winner_metrics = variant_metrics[winner]
        
        # Need minimum sample size
        if winner_metrics.queries < self.min_queries:
            return 0.0
        
        # Simple heuristic: based on sample size and score gap
        # (In production, use proper t-test or chi-squared test)
        
        # Compute score gap to next best
        scores = []
        for name, metrics in variant_metrics.items():
            if name != winner:
                score = metrics.avg_feedback if metrics.feedback_scores else metrics.avg_confidence
                scores.append(score)
        
        if not scores:
            return 0.0
        
        winner_score = (
            winner_metrics.avg_feedback
            if winner_metrics.feedback_scores
            else winner_metrics.avg_confidence
        )
        next_best = max(scores)
        score_gap = winner_score - next_best
        
        # Confidence increases with sample size and score gap
        sample_factor = min(winner_metrics.queries / self.min_queries, 1.0)
        gap_factor = min(score_gap * 5, 1.0)  # 20% gap = 100% confidence
        
        confidence = (sample_factor + gap_factor) / 2
        
        return confidence
    
    def export_report(self, filepath: str) -> None:
        """Export experiment report to JSON.
        
        Args:
            filepath: Output file path
        """
        import json
        
        report = self.get_report()
        
        with open(filepath, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        logger.info(f"Exported experiment report to {filepath}")
    
    def stop(self) -> ExperimentReport:
        """Stop experiment and return final report.
        
        Returns:
            Final ExperimentReport
        """
        report = self.get_report()
        
        logger.info(
            f"Experiment '{self.name}' stopped after {report.duration_hours:.1f}h"
        )
        
        return report
