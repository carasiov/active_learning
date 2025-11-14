"""Metric collection and registries for experiments.

IMPORTANT: The metrics infrastructure has been moved to src/metrics/ at the
repository root to enable reuse across projects (experiments, dashboard, etc.).

This module re-exports the new location for backward compatibility while keeping
experiment-specific default metrics (defaults.py) local.

New code should import directly from:
    from metrics import MetricContext, collect_metrics, register_metric

Old code continues to work:
    from ..metrics import MetricContext, collect_metrics
"""

# Re-export infrastructure from new location
from metrics import (
    MetricContext,
    MetricResult,
    collect_metrics,
    register_metric,
)

# Import defaults to ensure default metrics are registered
from . import defaults as _defaults  # noqa: F401

__all__ = [
    "MetricContext",
    "MetricResult",
    "collect_metrics",
    "register_metric",
]
