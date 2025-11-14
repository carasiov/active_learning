"""Metric collection and registries for experiments.

IMPORTANT: The metrics infrastructure has been moved to src/metrics/ at the
repository root to enable reuse across projects (experiments, dashboard, etc.).

This module re-exports the infrastructure and ensures default metrics are
registered by importing them from the providers module.

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
from metrics.providers import defaults as _defaults  # noqa: F401

__all__ = [
    "MetricContext",
    "MetricResult",
    "collect_metrics",
    "register_metric",
]
