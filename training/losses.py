from __future__ import annotations

from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import optax

from configs.base import SSVAEConfig

def reconstruction_loss(x: jnp.ndarray, recon: jnp.ndarray, weight: float) -> jnp.ndarray:
    """Return the reconstruction loss (MSE) scaled by ``weight``."""
    diff = jnp.square(x - recon)
    if diff.ndim > 1:
        axes = tuple(range(1, diff.ndim))
        per_sample = jnp.mean(diff, axis=axes)
    else:
        per_sample = diff
    return weight * jnp.mean(per_sample)


def kl_divergence(z_mean: jnp.ndarray, z_log: jnp.ndarray, weight: float) -> jnp.ndarray:
    """Return the KL divergence term scaled by ``weight``."""
    kl = -0.5 * (1.0 + z_log - jnp.square(z_mean) - jnp.exp(z_log))
    return weight * jnp.mean(jnp.sum(kl, axis=1))


def classification_loss(logits: jnp.ndarray, labels: jnp.ndarray, weight: float) -> jnp.ndarray:
    """Return the classification loss averaged over labeled examples only."""
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


def _contrastive_loss_stub(z: jnp.ndarray, weight: float) -> jnp.ndarray:
    """Placeholder contrastive loss; returns a zero-valued array compatible with ``weight``."""
    return jnp.array(0.0, dtype=z.dtype) * weight


def compute_loss_and_metrics(
    params: Dict[str, Dict[str, jnp.ndarray]],
    batch_x: jnp.ndarray,
    batch_y: jnp.ndarray,
    model_apply_fn: Callable[..., Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]],
    config: SSVAEConfig,
    rng: jax.Array | None,
    *,
    training: bool,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Mirror ``SSVAE._loss_and_metrics`` using the pure loss helpers."""
    use_key = rng if training else None
    z_mean, z_log, z, recon, logits = model_apply_fn(
        params,
        batch_x,
        training=training,
        key=use_key,
    )

    rec_loss = reconstruction_loss(batch_x, recon, config.recon_weight)
    kl_loss = kl_divergence(z_mean, z_log, config.kl_weight)
    # Compute classification loss for metrics (unweighted) and for the objective (weighted).
    cls_loss_unweighted = classification_loss(logits, batch_y, 1.0)
    cls_loss_weighted = classification_loss(logits, batch_y, config.label_weight)

    if config.use_contrastive:
        contrastive = _contrastive_loss_stub(z, config.contrastive_weight)
    else:
        contrastive = jnp.array(0.0, dtype=rec_loss.dtype)

    total = rec_loss + kl_loss + cls_loss_weighted + contrastive
    metrics = {
        "loss": total,
        "reconstruction_loss": rec_loss,
        "kl_loss": kl_loss,
        "classification_loss": cls_loss_unweighted,
        "weighted_classification_loss": cls_loss_weighted,
        "contrastive_loss": contrastive,
    }
    return total, metrics
