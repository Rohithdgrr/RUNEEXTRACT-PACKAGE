"""
Budget enforcement for RAG pipelines — per-query and per-day cost/latency caps.

Usage::

    from runeextract.rag.budget import BudgetManager

    budget = BudgetManager(
        cost_per_query=0.05,
        cost_per_day=50.0,
        latency_p95_ms=1500,
        on_exceeded="degrade",
    )

    # Before each query:
    if budget.can_query(estimated_cost=0.02):
        result = rag.query(...)
        budget.record(result.cost, result.latency_ms)
    else:
        result = budget.degraded_response("query")
"""

import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable


logger = logging.getLogger(__name__)


class DegradeAction(Enum):
    RAISE = "raise"
    FALLBACK = "fallback"
    DEGRADE = "degrade"
    CACHE_ONLY = "cache_only"


@dataclass
class BudgetConfig:
    """Budget limits for a RAG pipeline.

    Args:
        cost_per_query: Hard cap on cost per query (USD).
        cost_per_day: Hard cap on total cost per day (USD).
        latency_p95_ms: Soft cap — if exceeded, auto-degrades.
        tokens_per_query: Max tokens allowed per query.
        max_queries_per_minute: Rate limit.
        on_exceeded: Action when budget is exceeded.
            ``"raise"`` — raise BudgetExceededError.
            ``"fallback"`` — fall back to cache-only or keyword mode.
            ``"degrade"`` — skip reranker, use faster LLM.
    """
    cost_per_query: Optional[float] = None
    cost_per_day: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    tokens_per_query: Optional[int] = None
    max_queries_per_minute: Optional[int] = None
    on_exceeded: str = "degrade"

    @classmethod
    def from_env(cls) -> "BudgetConfig":
        return cls(
            cost_per_query=_env_float("RUNEEXTRACT_BUDGET_COST_PER_QUERY"),
            cost_per_day=_env_float("RUNEEXTRACT_BUDGET_COST_PER_DAY"),
            latency_p95_ms=_env_float("RUNEEXTRACT_BUDGET_LATENCY_P95_MS"),
            tokens_per_query=_env_int("RUNEEXTRACT_BUDGET_TOKENS_PER_QUERY"),
            max_queries_per_minute=_env_int("RUNEEXTRACT_BUDGET_MAX_QPM"),
            on_exceeded=os.environ.get("RUNEEXTRACT_BUDGET_ON_EXCEEDED", "degrade"),
        )


@dataclass
class BudgetState:
    """Current budget state — tracks usage and degradation level."""
    total_cost: float = 0.0
    daily_cost: float = 0.0
    query_count: int = 0
    latency_ms: List[float] = field(default_factory=list)
    degradation_level: int = 0  # 0 = full, 1 = no-reranker, 2 = cache-only, 3 = refuse
    last_reset_day: str = ""  # YYYY-MM-DD


class BudgetExceededError(Exception):
    """Raised when a budget limit is hit and action is 'raise'."""


class BudgetManager:
    """Enforce cost and latency budgets for RAG queries.

    Tracks costs across queries, auto-detects when limits are hit,
    and applies configurable degradation strategies.
    """

    def __init__(self, config: Optional[BudgetConfig] = None,
                 webhook: Optional[Callable[[str], None]] = None):
        self._config = config or BudgetConfig()
        self._state = BudgetState()
        self._recent_times: deque = deque(maxlen=1000)
        self._webhook = webhook
        self._degradation_actions: Dict[int, Dict[str, bool]] = {
            0: {"skip_reranker": False, "skip_hyde": False, "short_answer": False, "cache_only": False},
            1: {"skip_reranker": True, "skip_hyde": False, "short_answer": False, "cache_only": False},
            2: {"skip_reranker": True, "skip_hyde": True, "short_answer": True, "cache_only": False},
            3: {"skip_reranker": True, "skip_hyde": True, "short_answer": True, "cache_only": True},
        }
        self._reset_daily()

    def _reset_daily(self):
        today = datetime.now().strftime("%Y-%m-%d")
        if self._state.last_reset_day != today:
            self._state.daily_cost = 0.0
            self._state.last_reset_day = today

    # ---- Public API ----

    def can_query(self, estimated_cost: float = 0.0,
                  estimated_tokens: int = 0) -> Tuple[bool, Optional[str]]:
        """Check if a query can proceed given current budget state.

        Args:
            estimated_cost: Expected cost of this query (USD).
            estimated_tokens: Expected token count.

        Returns:
            Tuple of (allowed: bool, reason: Optional[str]).
        """
        self._reset_daily()

        if self._state.degradation_level >= 3:
            return False, "Hard budget cap reached — queries refused"

        if self._config.cost_per_query and estimated_cost > self._config.cost_per_query:
            return False, f"Estimated cost ${estimated_cost:.4f} exceeds per-query limit ${self._config.cost_per_query:.4f}"

        if self._config.cost_per_day and (self._state.daily_cost + estimated_cost) > self._config.cost_per_day:
            return False, f"Daily cost ${self._state.daily_cost:.4f} + ${estimated_cost:.4f} exceeds limit ${self._config.cost_per_day:.4f}"

        if self._config.max_queries_per_minute:
            cutoff = time.time() - 60
            recent = sum(1 for t in self._recent_times if t >= cutoff)
            if recent >= self._config.max_queries_per_minute:
                return False, f"Rate limit: {recent} queries in last minute (max {self._config.max_queries_per_minute})"

        return True, None

    def record(self, cost: float = 0.0, latency_ms: float = 0.0, tokens: int = 0):
        """Record a completed query's cost and latency.

        Updates budget state and adjusts degradation level if limits exceeded.
        """
        self._state.total_cost += cost
        self._state.daily_cost += cost
        self._state.query_count += 1
        self._state.latency_ms.append(latency_ms)
        self._recent_times.append(time.time())

        # Check for degradation triggers
        if self._config.cost_per_day and self._state.daily_cost > self._config.cost_per_day * 0.8:
            self._degrade(1, "Daily cost at 80% of limit")

        if self._config.cost_per_query and cost > self._config.cost_per_query:
            self._degrade(2, f"Query cost ${cost:.4f} exceeded per-query limit")

        if self._config.max_queries_per_minute:
            cutoff = time.time() - 60
            recent = sum(1 for t in self._recent_times if t >= cutoff)
            if recent >= self._config.max_queries_per_minute * 0.8:
                self._degrade(1, "Approaching rate limit")

    def get_state(self) -> BudgetState:
        return self._state

    def degradation_flags(self) -> Dict[str, bool]:
        """Return the active degradation flags for the current level."""
        level = min(self._state.degradation_level, 3)
        return dict(self._degradation_actions[level])

    def degrade_response(self, query: str) -> str:
        """Return a degradation notice when queries are refused."""
        level = self._state.degradation_level
        if level >= 3:
            return "Budget limit reached. Please try again later or reduce usage."
        return "Query degraded due to budget limits."

    def _degrade(self, level: int, reason: str):
        if level > self._state.degradation_level:
            self._state.degradation_level = level
            msg = f"Budget degradation: level {level} — {reason}"
            logger.warning(msg)
            if self._webhook:
                try:
                    self._webhook(msg)
                except Exception:
                    pass
            if self._config.on_exceeded == "raise":
                raise BudgetExceededError(reason)


def _env_float(key: str) -> Optional[float]:
    val = os.environ.get(key)
    if val is not None:
        try:
            return float(val)
        except ValueError:
            return None
    return None


def _env_int(key: str) -> Optional[int]:
    val = os.environ.get(key)
    if val is not None:
        try:
            return int(val)
        except ValueError:
            return None
    return None
