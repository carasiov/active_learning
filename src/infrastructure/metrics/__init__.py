"""Metric collection and registries for experiments.

This module provides infrastructure for computing and organizing experiment metrics
through a registry pattern. Metrics can report success, disabled, skipped, or failed
status using ComponentResult from common.status.

Architecture:
- registry.py: MetricContext, collect_metrics(), @register_metric decorator
- schema.py: Canonical naming constants for all metrics (LossKeys, MixtureKeys, etc.)

Usage:
    from metrics import MetricContext, collect_metrics, register_metric
    from metrics.schema import LossKeys, MixtureKeys

    @register_metric
    def my_metric_provider(context: MetricContext) -> ComponentResult:
        return ComponentResult.success(data={
            MixtureKeys.K_EFF: 7.3
        })

    # Later, in training pipeline:
    summary = collect_metrics(MetricContext(model, config, ...))
"""

from .registry import (
    MetricContext,
    MetricResult,
    collect_metrics,
    register_metric,
)

__all__ = [
    "MetricContext",
    "MetricResult",
    "collect_metrics",
    "register_metric",
]
