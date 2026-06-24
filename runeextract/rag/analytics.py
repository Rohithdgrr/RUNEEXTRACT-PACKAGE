"""
Feature 8: Auto-Generated Dashboards & Analytics

Real-time analytics dashboard with zero configuration.
Production visibility from day 1.
"""

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, DefaultDict, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query: str
    latency_ms: float
    confidence: float
    cost: float
    citations: int
    chunks_retrieved: int
    timestamp: float
    error: Optional[str] = None


@dataclass
class AnalyticsSummary:
    """Summary of RAG analytics."""
    total_queries: int = 0
    total_cost: float = 0.0
    avg_latency_ms: float = 0.0
    avg_confidence: float = 0.0
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0
    
    # Time-based metrics
    queries_last_hour: int = 0
    queries_last_day: int = 0
    cost_last_hour: float = 0.0
    cost_last_day: float = 0.0
    
    # Top items
    top_queries: List[Tuple[str, int]] = field(default_factory=list)
    top_documents: List[Tuple[str, int]] = field(default_factory=list)
    error_types: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_queries": self.total_queries,
            "total_cost": round(self.total_cost, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "avg_confidence": round(self.avg_confidence, 3),
            "error_rate": round(self.error_rate, 3),
            "cache_hit_rate": round(self.cache_hit_rate, 3),
            "queries_last_hour": self.queries_last_hour,
            "queries_last_day": self.queries_last_day,
            "cost_last_hour": round(self.cost_last_hour, 4),
            "cost_last_day": round(self.cost_last_day, 4),
            "top_queries": self.top_queries[:10],
            "top_documents": self.top_documents[:10],
            "error_types": self.error_types
        }
    
    def __str__(self) -> str:
        """Pretty print summary."""
        return f"""
RAG Analytics Summary
═══════════════════════════════════════════
Queries:      {self.total_queries:,}
Cost:         ${self.total_cost:.2f}
Avg Latency:  {self.avg_latency_ms:.0f}ms
Avg Confidence: {self.avg_confidence:.2%}
Error Rate:   {self.error_rate:.2%}
Cache Hit Rate: {self.cache_hit_rate:.2%}

Last Hour:    {self.queries_last_hour} queries, ${self.cost_last_hour:.2f}
Last Day:     {self.queries_last_day} queries, ${self.cost_last_day:.2f}

Top Queries:
{self._format_top_items(self.top_queries, 5)}

Top Documents:
{self._format_top_items(self.top_documents, 5)}
"""
    
    def _format_top_items(self, items: List[Tuple[str, int]], limit: int) -> str:
        """Format top items list."""
        lines = []
        for item, count in items[:limit]:
            lines.append(f"  {count:3d}x  {item[:60]}")
        return "\n".join(lines) if lines else "  (none)"


class RAGAnalytics:
    """Real-time analytics for RAG systems.
    
    Features:
    - Query volume & latency tracking
    - Cost monitoring
    - Confidence distribution
    - Top queries & documents
    - Error tracking
    - Time-series metrics
    
    Usage::
    
        analytics = RAGAnalytics()
        
        # Record query
        analytics.record_query(
            query="What is the policy?",
            latency_ms=847,
            confidence=0.87,
            cost=0.023,
            citations=3,
            chunks_retrieved=5
        )
        
        # Get summary
        summary = analytics.get_summary()
        print(summary)
        
        # Export data
        analytics.export_json("analytics.json")
    """
    
    def __init__(
        self,
        history_size: int = 10000,
        enable_time_series: bool = True
    ):
        """Initialize analytics tracker.
        
        Args:
            history_size: Max number of queries to keep in memory
            enable_time_series: Track time-series data for trends
        """
        self.history_size = history_size
        self.enable_time_series = enable_time_series
        
        # Query history
        self._queries: Deque[QueryMetrics] = deque(maxlen=history_size)
        
        # Aggregated metrics
        self._total_queries = 0
        self._total_cost = 0.0
        self._total_latency = 0.0
        self._total_confidence = 0.0
        self._error_count = 0
        self._cache_hits = 0
        
        # Frequency counters
        self._query_counts: DefaultDict[str, int] = defaultdict(int)
        self._document_counts: DefaultDict[str, int] = defaultdict(int)
        self._error_types: DefaultDict[str, int] = defaultdict(int)
        
        # Time-series data (hour buckets)
        if enable_time_series:
            self._hourly_metrics: DefaultDict[str, Dict[str, float]] = defaultdict(
                lambda: {"queries": 0, "cost": 0.0, "latency": 0.0, "confidence": 0.0}
            )
        
        logger.info(f"Analytics initialized (history_size={history_size})")
    
    def record_query(
        self,
        query: str,
        latency_ms: float,
        confidence: float,
        cost: float,
        citations: int,
        chunks_retrieved: int,
        document_sources: Optional[List[str]] = None,
        error: Optional[str] = None,
        cached: bool = False
    ) -> None:
        """Record query metrics.
        
        Args:
            query: Query text
            latency_ms: Query latency in milliseconds
            confidence: Confidence score (0-1)
            cost: Query cost in dollars
            citations: Number of citations
            chunks_retrieved: Number of chunks retrieved
            document_sources: List of source documents accessed
            error: Error message if query failed
            cached: Whether result was served from cache
        """
        timestamp = time.time()
        
        # Record metrics
        metrics = QueryMetrics(
            query=query[:100],  # Truncate for storage
            latency_ms=latency_ms,
            confidence=confidence,
            cost=cost,
            citations=citations,
            chunks_retrieved=chunks_retrieved,
            timestamp=timestamp,
            error=error
        )
        
        self._queries.append(metrics)
        
        # Update aggregates
        self._total_queries += 1
        self._total_cost += cost
        self._total_latency += latency_ms
        self._total_confidence += confidence
        
        if error:
            self._error_count += 1
            self._error_types[error] += 1
        
        if cached:
            self._cache_hits += 1
        
        # Update frequency counters
        self._query_counts[query[:50]] += 1
        
        if document_sources:
            for doc in document_sources:
                self._document_counts[doc] += 1
        
        # Update time-series
        if self.enable_time_series:
            hour_key = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:00")
            self._hourly_metrics[hour_key]["queries"] += 1
            self._hourly_metrics[hour_key]["cost"] += cost
            self._hourly_metrics[hour_key]["latency"] += latency_ms
            self._hourly_metrics[hour_key]["confidence"] += confidence
    
    def get_summary(self) -> AnalyticsSummary:
        """Get analytics summary."""
        summary = AnalyticsSummary()
        
        if self._total_queries == 0:
            return summary
        
        # Basic metrics
        summary.total_queries = self._total_queries
        summary.total_cost = self._total_cost
        summary.avg_latency_ms = self._total_latency / self._total_queries
        summary.avg_confidence = self._total_confidence / self._total_queries
        summary.error_rate = self._error_count / self._total_queries
        summary.cache_hit_rate = self._cache_hits / self._total_queries
        
        # Time-based metrics
        now = time.time()
        one_hour_ago = now - 3600
        one_day_ago = now - 86400
        
        for q in self._queries:
            if q.timestamp >= one_hour_ago:
                summary.queries_last_hour += 1
                summary.cost_last_hour += q.cost
            if q.timestamp >= one_day_ago:
                summary.queries_last_day += 1
                summary.cost_last_day += q.cost
        
        # Top queries
        summary.top_queries = sorted(
            self._query_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Top documents
        summary.top_documents = sorted(
            self._document_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Error types
        summary.error_types = dict(self._error_types)
        
        return summary
    
    def get_time_series(
        self,
        metric: str = "queries",
        hours: int = 24
    ) -> List[Tuple[str, float]]:
        """Get time-series data for a metric.
        
        Args:
            metric: Metric name ("queries", "cost", "latency", "confidence")
            hours: Number of hours to return
        
        Returns:
            List of (timestamp, value) tuples
        """
        if not self.enable_time_series:
            return []
        
        # Get last N hours
        now = datetime.now()
        data = []
        
        for i in range(hours):
            hour = now - timedelta(hours=i)
            hour_key = hour.strftime("%Y-%m-%d %H:00")
            
            if hour_key in self._hourly_metrics:
                value = self._hourly_metrics[hour_key].get(metric, 0.0)
                
                # Normalize averages
                if metric in ["latency", "confidence"]:
                    queries = self._hourly_metrics[hour_key]["queries"]
                    value = value / queries if queries > 0 else 0.0
                
                data.append((hour_key, value))
            else:
                data.append((hour_key, 0.0))
        
        return list(reversed(data))
    
    def get_confidence_distribution(self, bins: int = 10) -> Dict[str, int]:
        """Get distribution of confidence scores.
        
        Args:
            bins: Number of bins (0-10%, 10-20%, etc.)
        
        Returns:
            Dict of bin -> count
        """
        distribution: DefaultDict[str, int] = defaultdict(int)
        
        for q in self._queries:
            if q.error:
                continue
            
            bin_idx = min(int(q.confidence * bins), bins - 1)
            bin_label = f"{bin_idx * (100 // bins)}-{(bin_idx + 1) * (100 // bins)}%"
            distribution[bin_label] += 1
        
        return dict(distribution)
    
    def get_latency_percentiles(self) -> Dict[str, float]:
        """Get latency percentiles (p50, p95, p99)."""
        if not self._queries:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        
        latencies = sorted([q.latency_ms for q in self._queries if not q.error])
        
        if not latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        
        def percentile(data: List[float], p: float) -> float:
            idx = int(len(data) * p)
            return data[min(idx, len(data) - 1)]
        
        return {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99)
        }
    
    def export_json(self, filepath: str) -> None:
        """Export analytics to JSON file.
        
        Args:
            filepath: Output file path
        """
        data = {
            "summary": self.get_summary().to_dict(),
            "confidence_distribution": self.get_confidence_distribution(),
            "latency_percentiles": self.get_latency_percentiles(),
            "time_series": {
                "queries": self.get_time_series("queries", hours=24),
                "cost": self.get_time_series("cost", hours=24),
                "latency": self.get_time_series("latency", hours=24)
            },
            "recent_queries": [
                {
                    "query": q.query,
                    "latency_ms": q.latency_ms,
                    "confidence": q.confidence,
                    "cost": q.cost,
                    "timestamp": q.timestamp,
                    "error": q.error
                }
                for q in list(self._queries)[-100:]
            ]
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Analytics exported to {filepath}")
    
    def export_csv(self, filepath: str) -> None:
        """Export query history to CSV.
        
        Args:
            filepath: Output file path
        """
        import csv
        
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "query", "latency_ms", "confidence",
                "cost", "citations", "chunks", "error"
            ])
            
            for q in self._queries:
                writer.writerow([
                    datetime.fromtimestamp(q.timestamp).isoformat(),
                    q.query,
                    q.latency_ms,
                    q.confidence,
                    q.cost,
                    q.citations,
                    q.chunks_retrieved,
                    q.error or ""
                ])
        
        logger.info(f"Query history exported to {filepath}")
    
    def clear(self) -> None:
        """Clear all analytics data."""
        self._queries.clear()
        self._total_queries = 0
        self._total_cost = 0.0
        self._total_latency = 0.0
        self._total_confidence = 0.0
        self._error_count = 0
        self._cache_hits = 0
        self._query_counts.clear()
        self._document_counts.clear()
        self._error_types.clear()
        
        if self.enable_time_series:
            self._hourly_metrics.clear()
        
        logger.info("Analytics cleared")
