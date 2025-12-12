"""Curriculum trigger utilities for intelligent channel unlocking.

This module provides trigger-based unlock logic as an alternative to epoch-based
schedules. Unlock occurs when training plateaus AND latent distributions look
approximately normal.

Stage 4 of channel unlocking curriculum implementation.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import jax.numpy as jnp


def compute_normality_score(
    z_mean: jnp.ndarray,
    z_log_var: jnp.ndarray,
    k_active: int,
    latent_layout: str = "shared",
) -> float:
    """Compute latent normality proxy score.

    Measures how close the posterior q(z|x) is to N(0,I) over active channels.
    Score of 0 means perfect normality; higher means less normal.

    For each active channel k, computes:
        S_k = mean(||μ_k||²) + mean(|exp(logvar_k) - 1|²)

    Overall score: S = mean over active channels of S_k

    Args:
        z_mean: Posterior means. Shape depends on latent_layout:
            - "shared": [batch, latent_dim] (same latent for all components)
            - "decentralized": [batch, K, latent_dim] (per-component latents)
        z_log_var: Posterior log-variances, same shape as z_mean.
        k_active: Number of active channels (from curriculum).
        latent_layout: "shared" or "decentralized".

    Returns:
        Normality score (float). Lower is better (closer to N(0,I)).
    """
    if latent_layout == "decentralized" and z_mean.ndim == 3:
        # Decentralized: z_mean is [B, K, D], z_log_var is [B, K, D]
        # Only consider active channels (k < k_active)
        z_mean_active = z_mean[:, :k_active, :]  # [B, k_active, D]
        z_log_var_active = z_log_var[:, :k_active, :]  # [B, k_active, D]

        # Compute per-channel scores
        # S_k = mean over (batch, D) of (μ² + (exp(logvar) - 1)²)
        mean_sq = jnp.mean(z_mean_active ** 2, axis=(0, 2))  # [k_active]
        var = jnp.exp(z_log_var_active)
        var_deviation = jnp.mean((var - 1.0) ** 2, axis=(0, 2))  # [k_active]
        per_channel_score = mean_sq + var_deviation  # [k_active]

        # Average over active channels
        score = float(jnp.mean(per_channel_score))
    else:
        # Shared layout: z_mean is [B, D], z_log_var is [B, D]
        # Single score for the shared latent (all active channels use same latent)
        mean_sq = jnp.mean(z_mean ** 2)
        var = jnp.exp(z_log_var)
        var_deviation = jnp.mean((var - 1.0) ** 2)
        score = float(mean_sq + var_deviation)

    return score


def check_plateau(
    history: Dict[str, List[float]],
    metric: str,
    window: int,
    min_improvement: float,
) -> Tuple[bool, float]:
    """Check if training has plateaued based on metric history.

    Plateau is detected when the relative improvement over the window is below
    the threshold.

    Args:
        history: Training history dict with metric lists.
        metric: Name of metric to check (e.g., "reconstruction_loss").
        window: Number of epochs to look back.
        min_improvement: Minimum relative improvement required to NOT be a plateau.
            Computed as: (old - new) / |old|

    Returns:
        Tuple of (is_plateau: bool, improvement: float).
        is_plateau is True if improvement < min_improvement.
    """
    if metric not in history:
        return False, 0.0

    values = history[metric]
    if len(values) < window:
        # Not enough history yet
        return False, 0.0

    # Get values at start and end of window
    old_value = values[-window]
    new_value = values[-1]

    # Avoid division by zero
    if abs(old_value) < 1e-10:
        if abs(new_value) < 1e-10:
            # Both near zero - treat as plateau
            return True, 0.0
        else:
            # Old was zero, new is not - significant change
            return False, 1.0

    # Relative improvement (positive if loss decreased)
    improvement = (old_value - new_value) / abs(old_value)

    is_plateau = improvement < min_improvement
    return is_plateau, float(improvement)


def should_unlock_trigger(
    history: Dict[str, List[float]],
    normality_score: float,
    config,
    current_k_active: int,
) -> Tuple[bool, Dict[str, float]]:
    """Determine if a trigger-based unlock should occur.

    Unlock triggers when BOTH conditions are met:
    1. Training has plateaued (recon loss improvement below threshold)
    2. Latent normality score is below threshold (active channels are "settled")

    Args:
        history: Training history dict.
        normality_score: Current latent normality score.
        config: SSVAEConfig with trigger parameters.
        current_k_active: Current number of active channels.

    Returns:
        Tuple of (should_unlock: bool, diagnostics: dict).
        diagnostics contains: plateau_detected, plateau_improvement, normality_score,
        normality_ok, should_unlock.
    """
    # Check if already at max
    if current_k_active >= config.curriculum_max_k_active:
        return False, {
            "plateau_detected": False,
            "plateau_improvement": 0.0,
            "normality_score": normality_score,
            "normality_ok": True,
            "should_unlock": False,
            "at_max": True,
        }

    # Check plateau condition
    is_plateau, improvement = check_plateau(
        history,
        config.curriculum_plateau_metric,
        config.curriculum_plateau_window_epochs,
        config.curriculum_plateau_min_improvement,
    )

    # Check normality condition
    normality_ok = normality_score < config.curriculum_normality_threshold

    # Unlock only if both conditions met
    should_unlock = is_plateau and normality_ok

    diagnostics = {
        "plateau_detected": is_plateau,
        "plateau_improvement": improvement,
        "normality_score": normality_score,
        "normality_ok": normality_ok,
        "should_unlock": should_unlock,
        "at_max": False,
    }

    return should_unlock, diagnostics


def compute_normality_score_from_extras(
    extras: Dict[str, jnp.ndarray],
    k_active: int,
    latent_layout: str = "shared",
) -> float:
    """Compute normality score from forward pass extras dict.

    Convenience wrapper that extracts z_mean and z_log_var from extras
    and calls compute_normality_score.

    Args:
        extras: Forward pass extras dict containing per-component or shared latents.
        k_active: Number of active channels.
        latent_layout: "shared" or "decentralized".

    Returns:
        Normality score (float).
    """
    if latent_layout == "decentralized":
        # Look for per-component latents
        z_mean = extras.get("z_mean_per_component")
        z_log_var = extras.get("z_log_var_per_component")
        if z_mean is None or z_log_var is None:
            # Fallback to shared latents if per-component not available
            z_mean = extras.get("z_mean")
            z_log_var = extras.get("z_log_var")
            if z_mean is None or z_log_var is None:
                return 0.0  # Can't compute without latent stats
            latent_layout = "shared"  # Use shared computation
    else:
        z_mean = extras.get("z_mean")
        z_log_var = extras.get("z_log_var")
        if z_mean is None or z_log_var is None:
            return 0.0  # Can't compute without latent stats

    return compute_normality_score(z_mean, z_log_var, k_active, latent_layout)
