"""
Loss computation with explicit PriorMode contracts (v2).

This version uses the PriorMode protocol for cleaner separation:
- Priors compute their own KL terms
- Priors handle reconstruction loss (weighted vs simple)
- No runtime type checking with hasattr()

Once validated, this will replace losses.py.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import optax

from ssvae.config import SSVAEConfig
from ssvae.priors.base import EncoderOutput, PriorMode


def compute_loss_and_metrics_v2(
    params: Dict[str, Dict[str, jnp.ndarray]],
    batch_x: jnp.ndarray,
    batch_y: jnp.ndarray,
    model_apply_fn: Callable,
    config: SSVAEConfig,
    prior: PriorMode,
    rng: jax.Array | None,
    *,
    training: bool,
    kl_c_scale: float = 1.0,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute loss and metrics using PriorMode abstraction.

    Args:
        params: Model parameters
        batch_x: Input images [batch, H, W]
        batch_y: Labels [batch] (NaN for unlabeled)
        model_apply_fn: Forward function
        config: Model configuration
        prior: Prior mode instance
        rng: Random key for sampling (None for deterministic)
        training: Whether in training mode
        kl_c_scale: Annealing factor for KL_c term

    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    # Forward pass
    use_key = rng if training else None
    forward_output = model_apply_fn(
        params,
        batch_x,
        training=training,
        key=use_key,
    )

    # Unpack forward output
    component_logits, z_mean, z_log, z, recon, class_logits, extras = forward_output

    # Create standardized encoder output
    encoder_output = EncoderOutput(
        z_mean=z_mean,
        z_log_var=z_log,
        z=z,
        component_logits=component_logits,
        extras=extras if hasattr(extras, "get") else None,
    )

    # Reconstruction loss (prior handles weighting for mixture)
    # For mixture: recon should be per-component [batch, K, H, W]
    # For standard: recon is simple [batch, H, W]
    if prior.requires_component_embeddings() and encoder_output.extras:
        # Mixture prior: use per-component reconstructions
        recon_per_component = encoder_output.extras.get("recon_per_component")
        if recon_per_component is not None:
            recon_loss = prior.compute_reconstruction_loss(
                batch_x, recon_per_component, encoder_output, config
            )
        else:
            # Fallback: use weighted reconstruction from model
            recon_loss = prior.compute_reconstruction_loss(
                batch_x, recon, encoder_output, config
            )
    else:
        # Standard prior: simple reconstruction
        recon_loss = prior.compute_reconstruction_loss(
            batch_x, recon, encoder_output, config
        )

    # KL divergence and regularization terms from prior
    kl_terms = prior.compute_kl_terms(encoder_output, config)

    # Apply KL_c annealing if present
    if "kl_c" in kl_terms:
        kl_terms["kl_c"] = kl_terms["kl_c"] * kl_c_scale

    # Classification loss (prior-agnostic)
    cls_loss_unweighted = _classification_loss(class_logits, batch_y, weight=1.0)
    cls_loss_weighted = _classification_loss(class_logits, batch_y, weight=config.label_weight)

    # Assemble total loss
    total_kl = sum(
        v for k, v in kl_terms.items()
        if k in ("kl_z", "kl_c", "dirichlet_penalty", "usage_sparsity")
    )
    total = recon_loss + total_kl + cls_loss_weighted

    # Build metrics dictionary
    metrics = {
        "loss": total,
        "reconstruction_loss": recon_loss,
        "classification_loss": cls_loss_unweighted,
        "weighted_classification_loss": cls_loss_weighted,
    }

    # Add all KL terms from prior
    for key, value in kl_terms.items():
        metrics[key] = value

    # Aggregate kl_loss for backward compatibility
    metrics["kl_loss"] = metrics.get("kl_z", 0.0) + metrics.get("kl_c", 0.0)

    # Loss without global regularizers (for monitoring)
    metrics["loss_no_global_priors"] = (
        recon_loss
        + metrics.get("kl_z", 0.0)
        + metrics.get("kl_c", 0.0)
        + cls_loss_weighted
    )

    # Add contrastive loss placeholder for backward compatibility
    metrics["contrastive_loss"] = jnp.array(0.0, dtype=recon_loss.dtype)

    return total, metrics


def _classification_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    weight: float,
) -> jnp.ndarray:
    """Compute classification loss over labeled examples only.

    Args:
        logits: Class logits [batch, num_classes]
        labels: Labels [batch] (NaN for unlabeled)
        weight: Loss weight

    Returns:
        Weighted cross-entropy loss
    """
    labels = labels.reshape((-1,))
    mask = jnp.logical_not(jnp.isnan(labels))
    mask_float = mask.astype(jnp.float32)
    labels_int = jnp.where(mask, labels, 0.0).astype(jnp.int32)

    per_example_ce = optax.softmax_cross_entropy_with_integer_labels(logits, labels_int)
    masked_ce_sum = jnp.sum(per_example_ce * mask_float)
    labeled_count = jnp.sum(mask_float)
    zero = jnp.array(0.0, dtype=per_example_ce.dtype)

    mean_ce = jax.lax.cond(
        labeled_count > 0,
        lambda: masked_ce_sum / labeled_count,
        lambda: zero,
    )
    return weight * mean_ce
