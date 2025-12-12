"""Shared typing utilities for application layer."""
from __future__ import annotations

from typing import Dict, Protocol, Tuple

import jax
import jax.numpy as jnp

from .state import SSVAETrainState

MetricsDict = Dict[str, jnp.ndarray]


class TrainStepFn(Protocol):
    """Signature for train_step functions (supports optional τ context and curriculum)."""

    def __call__(
        self,
        state: SSVAETrainState,
        batch_x: jnp.ndarray,
        batch_y: jnp.ndarray,
        key: jax.Array,
        kl_c_scale: float,
        tau: jnp.ndarray | None = None,
        gumbel_temperature: float | None = None,
        k_active: int | None = None,
        use_straight_through: bool | None = None,
        effective_logit_mog_weight: float | None = None,
    ) -> Tuple[SSVAETrainState, MetricsDict]:
        ...


class EvalMetricsFn(Protocol):
    """Signature for eval metrics functions (supports optional τ context and curriculum)."""

    def __call__(
        self,
        params: Dict[str, Dict[str, jnp.ndarray]],
        batch_x: jnp.ndarray,
        batch_y: jnp.ndarray,
        tau: jnp.ndarray | None = None,
        k_active: int | None = None,
    ) -> MetricsDict:
        ...
