"""Simple registry for experiment metric providers."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, MutableMapping, Optional

import numpy as np

MetricResult = Dict[str, Any]
MetricProvider = Callable[["MetricContext"], Optional[MetricResult]]


@dataclass(slots=True)
class MetricContext:
    """Shared information passed to every metric provider."""

    model: Any
    config: Any
    history: Dict[str, List[float]]
    x_train: np.ndarray
    y_true: np.ndarray
    y_semi: np.ndarray
    latent: np.ndarray
    reconstructions: np.ndarray
    predictions: np.ndarray
    certainty: np.ndarray
    responsibilities: Optional[np.ndarray]
    pi_values: Optional[np.ndarray]
    train_time: float
    diagnostics_dir: Optional[Path]


_REGISTRY: List[MetricProvider] = []


def register_metric(func: MetricProvider) -> MetricProvider:
    """Register a metric provider function."""
    _REGISTRY.append(func)
    return func


def collect_metrics(context: MetricContext) -> MetricResult:
    """Run all registered metric providers and merge their results."""
    summary: MetricResult = {}
    for provider in _REGISTRY:
        result = provider(context)
        if not result:
            continue
        _deep_merge(summary, result)
    return summary


def _deep_merge(base: MutableMapping[str, Any], update: MutableMapping[str, Any]) -> None:
    for key, value in update.items():
        if (
            key in base
            and isinstance(base[key], MutableMapping)
            and isinstance(value, MutableMapping)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value


# Ensure default metrics are registered on import
from . import defaults as _defaults  # noqa: F401,E402
