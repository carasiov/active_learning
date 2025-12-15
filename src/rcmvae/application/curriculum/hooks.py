"""Training loop hooks for curriculum integration."""
from __future__ import annotations

from typing import Callable, Dict, Optional

import jax.numpy as jnp

from rcmvae.application.curriculum.controller import CurriculumController
from rcmvae.application.services.training_service import TrainerLoopHooks
from rcmvae.application.runtime.state import SSVAETrainState
from rcmvae.application.runtime.types import MetricsDict


def build_curriculum_hooks(
    controller: CurriculumController,
    on_unlock: Optional[Callable[[int, int, Dict], None]] = None,
) -> TrainerLoopHooks:
    """Build TrainerLoopHooks for curriculum learning.

    The hooks inject active_mask and optional gumbel_temperature override
    into the training step context, and handle epoch-end curriculum updates.

    Args:
        controller: The curriculum controller instance
        on_unlock: Optional callback called when an unlock occurs.
                   Signature: on_unlock(epoch, k_active, metrics)

    Returns:
        TrainerLoopHooks instance ready to be passed to Trainer.train()

    Example:
        >>> controller = CurriculumController(config, k_max=10)
        >>> hooks = build_curriculum_hooks(controller)
        >>> trainer.train(..., loop_hooks=hooks)
    """

    def batch_context_fn(
        state: SSVAETrainState,
        batch_x: jnp.ndarray,
        batch_y: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        """Provide curriculum context for each training batch."""
        del state, batch_x, batch_y  # Unused

        context = {
            "active_mask": jnp.array(controller.get_active_mask()),
        }

        # During kick window: use soft routing (disable ST) + high temperature
        # This allows the newly unlocked channel to receive responsibility mass
        # Under ST Gumbel, argmax is invariant to temperature, so we need soft routing
        if controller.is_in_kick():
            temp_override = controller.get_gumbel_temperature_override()
            if temp_override is not None:
                context["gumbel_temperature"] = jnp.array(temp_override)
            # Disable straight-through during kick: use soft routing
            # so temperature actually affects the routing distribution
            context["straight_through_gumbel"] = False

        return context

    def eval_context_fn() -> Dict[str, jnp.ndarray]:
        """Provide curriculum context for evaluation.

        During evaluation, we use the same active_mask but do NOT apply
        kick temperature (evaluation should be deterministic/standard).
        """
        return {
            "active_mask": jnp.array(controller.get_active_mask()),
        }

    def on_epoch_end_fn(epoch: int, metrics: Dict[str, float]) -> None:
        """Process end of epoch for curriculum updates.

        Calls the controller's on_epoch_end to check for unlocks
        and update curriculum state.
        """
        result = controller.on_epoch_end(epoch, metrics)

        # Call user's on_unlock callback if an unlock occurred
        if result.get("unlocked") and on_unlock is not None:
            on_unlock(epoch, result["k_active"], metrics)

    return TrainerLoopHooks(
        batch_context_fn=batch_context_fn,
        post_batch_fn=None,  # No per-batch updates needed
        eval_context_fn=eval_context_fn,
        on_epoch_end_fn=on_epoch_end_fn,
    )


def merge_hooks(
    hooks_list: list[TrainerLoopHooks | None],
) -> TrainerLoopHooks | None:
    """Merge multiple TrainerLoopHooks into a single instance.

    Context functions are chained: each returns a dict, and all dicts
    are merged (later hooks can override earlier ones).

    Args:
        hooks_list: List of hooks to merge (None entries are skipped)

    Returns:
        Merged TrainerLoopHooks, or None if all inputs were None
    """
    # Filter out None entries
    valid_hooks = [h for h in hooks_list if h is not None]
    if not valid_hooks:
        return None
    if len(valid_hooks) == 1:
        return valid_hooks[0]

    # Collect all functions
    batch_fns = [h.batch_context_fn for h in valid_hooks if h.batch_context_fn is not None]
    post_fns = [h.post_batch_fn for h in valid_hooks if h.post_batch_fn is not None]
    eval_fns = [h.eval_context_fn for h in valid_hooks if h.eval_context_fn is not None]
    epoch_end_fns = [h.on_epoch_end_fn for h in valid_hooks if h.on_epoch_end_fn is not None]

    def merged_batch_context_fn(
        state: SSVAETrainState,
        batch_x: jnp.ndarray,
        batch_y: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray] | None:
        result: Dict[str, jnp.ndarray] = {}
        for fn in batch_fns:
            ctx = fn(state, batch_x, batch_y)
            if ctx:
                result.update(ctx)
        return result if result else None

    def merged_post_batch_fn(
        state: SSVAETrainState,
        batch_x: jnp.ndarray,
        batch_y: jnp.ndarray,
        batch_metrics: MetricsDict,
    ) -> None:
        for fn in post_fns:
            fn(state, batch_x, batch_y, batch_metrics)

    def merged_eval_context_fn() -> Dict[str, jnp.ndarray] | None:
        result: Dict[str, jnp.ndarray] = {}
        for fn in eval_fns:
            ctx = fn()
            if ctx:
                result.update(ctx)
        return result if result else None

    def merged_on_epoch_end_fn(epoch: int, metrics: Dict[str, float]) -> None:
        for fn in epoch_end_fns:
            fn(epoch, metrics)

    return TrainerLoopHooks(
        batch_context_fn=merged_batch_context_fn if batch_fns else None,
        post_batch_fn=merged_post_batch_fn if post_fns else None,
        eval_context_fn=merged_eval_context_fn if eval_fns else None,
        on_epoch_end_fn=merged_on_epoch_end_fn if epoch_end_fns else None,
    )
