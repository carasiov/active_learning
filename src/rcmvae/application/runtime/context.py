"""Runtime container tying together network, state, and compiled functions."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import jax

from .state import SSVAETrainState
from .types import EvalMetricsFn, MetricsDict, TrainStepFn
from rcmvae.domain.network import SSVAENetwork
from rcmvae.domain.priors.base import PriorMode


@dataclass(frozen=True)
class ModelRuntime:
    """Immutable snapshot of the current model execution context."""

    network: SSVAENetwork
    state: SSVAETrainState
    train_step_fn: TrainStepFn
    eval_metrics_fn: EvalMetricsFn
    prior: PriorMode
    shuffle_rng: jax.Array

    def replace(self, **kwargs) -> "ModelRuntime":
        """Return a new runtime with selected fields updated."""
        return replace(self, **kwargs)
