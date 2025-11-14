"""Simple registry for experiment metric providers.

Supports ComponentResult status objects alongside
legacy dict/None returns for backward compatibility.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Union

import numpy as np

from infrastructure import ComponentResult

# Type aliases
MetricResult = Dict[str, Any]
MetricProvider = Callable[["MetricContext"], Union[ComponentResult, Optional[MetricResult]]]

logger = logging.getLogger("experiment.metrics")


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
    """Run all registered metric providers and merge their results.

    Handles both ComponentResult  and legacy dict/None returns
    for backward compatibility.

    Args:
        context: Metric computation context

    Returns:
        Nested dict with all metrics and status information

    Behavior:
    - ComponentResult.SUCCESS: Include data in summary
    - ComponentResult.DISABLED: Log info, include status in summary
    - ComponentResult.SKIPPED: Log warning, include status in summary
    - ComponentResult.FAILED: Log error, include status in summary
    - Dict: Legacy return, merge directly (backward compat)
    - None: Legacy return, skip silently (backward compat)
    """
    summary: MetricResult = {}

    for provider in _REGISTRY:
        provider_name = provider.__name__

        try:
            result = provider(context)
        except Exception as e:
            logger.error(
                f"Metric provider '{provider_name}' raised exception: {e}",
                exc_info=True,
            )
            # Record failure in summary
            summary[provider_name] = {
                "status": "failed",
                "reason": f"Exception: {str(e)}",
            }
            continue

        # Handle ComponentResult 
        if isinstance(result, ComponentResult):
            if result.is_success:
                logger.debug(f"✓ {provider_name}: {list(result.data.keys())}")
                _deep_merge(summary, result.data)

            elif result.is_disabled:
                logger.debug(f"○ {provider_name}: disabled ({result.reason})")
                summary[provider_name] = result.to_dict()

            elif result.is_skipped:
                logger.warning(f"⊘ {provider_name}: skipped ({result.reason})")
                summary[provider_name] = result.to_dict()

            elif result.is_failed:
                logger.error(
                    f"✗ {provider_name}: failed ({result.reason})",
                    exc_info=result.error,
                )
                summary[provider_name] = result.to_dict()

        # Handle legacy dict return (backward compat)
        elif isinstance(result, dict):
            logger.debug(f"✓ {provider_name}: {list(result.keys())} (legacy)")
            _deep_merge(summary, result)

        # Handle legacy None return (backward compat)
        elif result is None:
            logger.debug(f"○ {provider_name}: None (legacy)")
            # Skip silently for backward compatibility

        else:
            logger.error(
                f"Metric provider '{provider_name}' returned unexpected type: "
                f"{type(result)}. Expected ComponentResult, dict, or None."
            )

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


