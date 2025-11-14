from __future__ import annotations

from typing import Any

import jax
from flax.training import train_state


class SSVAETrainState(train_state.TrainState):
    """TrainState carrying RNG metadata required during training."""

    rng: jax.Array

    @classmethod
    def create(
        cls,
        *,
        apply_fn,
        params,
        tx,
        rng: jax.Array,
        **kwargs: Any,
    ) -> "SSVAETrainState":
        """Instantiate the training state with optimizer parameters and RNG."""
        return cls(
            step=kwargs.pop("step", 0),
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=tx.init(params),
            rng=rng,
            **kwargs,
        )
