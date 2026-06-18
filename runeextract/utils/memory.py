"""
Memory profiling utility for extraction operations.

Tracks memory usage before/after extraction, warns when thresholds
are exceeded, and provides context managers for scoped profiling.
"""

import logging
import os
import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Memory usage snapshot at a point in time."""
    rss_mb: float
    vms_mb: float
    timestamp: float = 0.0
    label: str = ""


@dataclass
class MemoryProfile:
    """Result of a memory profiling session."""
    before: MemorySnapshot = field(default_factory=lambda: MemorySnapshot(0, 0))
    after: MemorySnapshot = field(default_factory=lambda: MemorySnapshot(0, 0))
    peak_mb: float = 0.0
    diff_mb: float = 0.0
    warnings: list = field(default_factory=list)
    exceeded_limit: bool = False


def _get_process_memory() -> Optional[Dict[str, float]]:
    """Get current process memory usage in MB.

    Uses psutil if available; falls back to /proc/self/status on Linux
    or returns None on unsupported platforms.

    Returns:
        Dict with 'rss' and 'vms' in MB, or None if unavailable.
    """
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        mi = proc.memory_info()
        return {"rss": mi.rss / (1024 * 1024), "vms": mi.vms / (1024 * 1024)}
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: try /proc/self/status (Linux only)
    try:
        with open("/proc/self/status") as fh:
            data = fh.read()
        rss_kb = 0
        vms_kb = 0
        for line in data.splitlines():
            if line.startswith("VmRSS:"):
                rss_kb = int(line.split()[1])
            elif line.startswith("VmSize:"):
                vms_kb = int(line.split()[1])
        if rss_kb or vms_kb:
            return {"rss": rss_kb / 1024, "vms": vms_kb / 1024}
    except Exception:
        pass

    return None


def snapshot(label: str = "") -> MemorySnapshot:
    """Take a memory snapshot at the current moment.

    Args:
        label: Optional label for the snapshot

    Returns:
        MemorySnapshot with available metrics (0 values if unavailable).
    """
    import time
    mem = _get_process_memory()
    if mem is None:
        return MemorySnapshot(rss_mb=0.0, vms_mb=0.0, timestamp=time.time(), label=label)
    return MemorySnapshot(
        rss_mb=mem["rss"],
        vms_mb=mem["vms"],
        timestamp=time.time(),
        label=label,
    )


class MemoryProfiler:
    """Profile memory usage of extraction operations.

    Args:
        warn_mb: Memory threshold in MB for warnings (default: 500)
        limit_mb: Hard memory limit in MB (default: 0 = no limit)
        enabled: Whether profiling is enabled (default: True)
    """

    def __init__(
        self,
        warn_mb: float = 500.0,
        limit_mb: float = 0.0,
        enabled: bool = True,
    ):
        self.warn_mb = warn_mb
        self.limit_mb = limit_mb
        self.enabled = enabled
        self._peak: float = 0.0
        self._lock = threading.Lock()

    def profile(self, label: str = "") -> MemoryProfile:
        """Run a profiling session using a context manager.

        Usage:
            profiler = MemoryProfiler(warn_mb=300)
            with profiler.profile("extract pdf"):
                doc = extract(...)
            print(profiler.result)

        Returns:
            MemoryProfile context manager.
        """
        return _ProfileContext(self, label)

    def get_peak_mb(self) -> float:
        """Return the peak RSS memory observed in MB."""
        return self._peak

    def reset(self):
        """Reset peak memory tracking."""
        self._peak = 0.0

    def _check(self, current_mb: float, profile: MemoryProfile):
        """Check memory against thresholds and record warnings."""
        with self._lock:
            if current_mb > self._peak:
                self._peak = current_mb

        profile.peak_mb = max(profile.peak_mb, current_mb)

        if self.warn_mb > 0 and current_mb > self.warn_mb:
            warn = f"Memory warning: {current_mb:.1f} MB exceeds threshold ({self.warn_mb:.1f} MB)"
            logger.warning(warn)
            profile.warnings.append(warn)

        if self.limit_mb > 0 and current_mb > self.limit_mb:
            profile.exceeded_limit = True


class _ProfileContext:
    """Context manager returned by MemoryProfiler.profile()."""

    def __init__(self, profiler: MemoryProfiler, label: str):
        self.profiler = profiler
        self.label = label
        self.result = MemoryProfile()

    def __enter__(self) -> MemoryProfile:
        self.result.before = snapshot(f"before: {self.label}")
        return self.result

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.result.after = snapshot(f"after: {self.label}")
        self.result.diff_mb = self.result.after.rss_mb - self.result.before.rss_mb
        self.profiler._check(self.result.after.rss_mb, self.result)
