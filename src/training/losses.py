from __future__ import annotations

from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import optax

from ssvae.config import SSVAEConfig


EPS = 1e-8


def reconstruction_loss_mse(x: jnp.ndarray, recon: jnp.ndarray, weight: float) -> jnp.ndarray:
    """Mean squared error reconstruction loss for standard prior."""
    diff = jnp.square(x - recon)
    if diff.ndim > 1:
        axes = tuple(range(1, diff.ndim))
        per_sample = jnp.mean(diff, axis=axes)
    else:
        per_sample = diff
    return weight * jnp.mean(per_sample)


def reconstruction_loss_bce(x: jnp.ndarray, logits: jnp.ndarray, weight: float) -> jnp.ndarray:
    """Binary cross-entropy with logits (numerically stable) for standard prior."""
    if x.ndim > 2:
        x_flat = x.reshape((x.shape[0], -1))
        logits_flat = logits.reshape((logits.shape[0], -1))
    else:
        x_flat = x
        logits_flat = logits

    max_val = jnp.maximum(logits_flat, 0.0)
    per_pixel_loss = max_val - x_flat * logits_flat + jnp.log1p(jnp.exp(-jnp.abs(logits_flat)))
    per_sample_loss = jnp.sum(per_pixel_loss, axis=1)
    batch_loss = jnp.mean(per_sample_loss)
    return weight * batch_loss


def weighted_reconstruction_loss_mse(
    x: jnp.ndarray,
    recon_components: jnp.ndarray,
    responsibilities: jnp.ndarray,
    weight: float,
) -> jnp.ndarray:
    """Expected reconstruction MSE under q(c|x)."""
    diff = jnp.square(x[:, None, ...] - recon_components)
    axes = tuple(range(2, diff.ndim))
    per_component = jnp.mean(diff, axis=axes)
    weighted = jnp.sum(responsibilities * per_component, axis=1)
    return weight * jnp.mean(weighted)


def weighted_reconstruction_loss_bce(
    x: jnp.ndarray,
    logits_components: jnp.ndarray,
    responsibilities: jnp.ndarray,
    weight: float,
) -> jnp.ndarray:
    """Expected BCE reconstruction loss under q(c|x)."""
    if x.ndim > 2:
        x_flat = x.reshape((x.shape[0], -1))
        logits_flat = logits_components.reshape((logits_components.shape[0], logits_components.shape[1], -1))
    else:
        x_flat = x
        logits_flat = logits_components

    max_val = jnp.maximum(logits_flat, 0.0)
    per_pixel_loss = max_val - x_flat[:, None, :] * logits_flat + jnp.log1p(jnp.exp(-jnp.abs(logits_flat)))
    per_component = jnp.sum(per_pixel_loss, axis=-1)
    weighted = jnp.sum(responsibilities * per_component, axis=1)
    return weight * jnp.mean(weighted)


def reconstruction_loss(
    x: jnp.ndarray,
    recon: jnp.ndarray,
    weight: float,
    loss_type: str = "mse",
) -> jnp.ndarray:
    """Compute reconstruction loss for standard prior modes."""
    if loss_type == "mse":
        return reconstruction_loss_mse(x, recon, weight)
    if loss_type == "bce":
        return reconstruction_loss_bce(x, recon, weight)
    raise ValueError(f"Unknown reconstruction_loss type: '{loss_type}'. Valid options: 'mse', 'bce'")


def weighted_reconstruction_loss(
    x: jnp.ndarray,
    recon_components: jnp.ndarray,
    responsibilities: jnp.ndarray,
    weight: float,
    loss_type: str,
) -> jnp.ndarray:
    """Compute expectation of reconstruction loss over mixture components."""
    if loss_type == "mse":
        return weighted_reconstruction_loss_mse(x, recon_components, responsibilities, weight)
    if loss_type == "bce":
        return weighted_reconstruction_loss_bce(x, recon_components, responsibilities, weight)
    raise ValueError(f"Unknown reconstruction_loss type: '{loss_type}'. Valid options: 'mse', 'bce'")


def kl_divergence(z_mean: jnp.ndarray, z_log: jnp.ndarray, weight: float) -> jnp.ndarray:
    """Return the KL divergence term scaled by ``weight``."""
    kl = -0.5 * (1.0 + z_log - jnp.square(z_mean) - jnp.exp(z_log))
    return weight * jnp.mean(jnp.sum(kl, axis=1))


def categorical_kl(
    responsibilities: jnp.ndarray,
    pi: jnp.ndarray,
    weight: float,
) -> jnp.ndarray:
    """Compute KL(q(c|x) || π) averaged over the batch."""
    resp_safe = jnp.clip(responsibilities, EPS, 1.0)
    pi_safe = jnp.clip(pi, EPS, 1.0)
    log_ratio = jnp.log(resp_safe) - jnp.log(pi_safe)[None, :]
    per_sample = jnp.sum(resp_safe * log_ratio, axis=1)
    return weight * jnp.mean(per_sample)


def dirichlet_map_penalty(pi: jnp.ndarray, alpha: float | None, weight: float) -> jnp.ndarray:
    """Dirichlet MAP penalty on π; returns zero when alpha is None."""
    if alpha is None:
        return jnp.array(0.0, dtype=pi.dtype)
    pi_safe = jnp.clip(pi, EPS, 1.0)
    penalty = -(alpha - 1.0) * jnp.sum(jnp.log(pi_safe))
    return weight * penalty


def usage_sparsity_penalty(responsibilities: jnp.ndarray, weight: float) -> jnp.ndarray:
    """Usage sparsity penalty based on empirical component frequencies."""
    if weight == 0.0:
        return jnp.array(0.0, dtype=responsibilities.dtype)
    hat_p = jnp.mean(responsibilities, axis=0)
    penalty = jnp.sum(hat_p * jnp.log(jnp.clip(hat_p, EPS, 1.0)))
    return weight * penalty


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
   model_apply_fn: Callable[..., Tuple],
   config: SSVAEConfig,
   rng: jax.Array | None,
   *,
   training: bool,
    kl_c_scale: float = 1.0,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Mirror ``SSVAE._loss_and_metrics`` using the pure loss helpers."""
    use_key = rng if training else None
    forward_output = model_apply_fn(
        params,
        batch_x,
        training=training,
        key=use_key,
    )
    component_logits, z_mean, z_log, z, recon, logits, extras = forward_output

    if hasattr(extras, "get"):
        responsibilities = extras.get("responsibilities")
        recon_components = extras.get("recon_per_component")
        pi = extras.get("pi")
    else:
        responsibilities = recon_components = pi = None

    if responsibilities is not None and recon_components is not None and pi is not None:
        rec_loss = weighted_reconstruction_loss(
            batch_x,
            recon_components,
            responsibilities,
            config.recon_weight,
            loss_type=config.reconstruction_loss,
        )
        kl_z = kl_divergence(z_mean, z_log, config.kl_weight)
        kl_c_weight = config.kl_c_weight * kl_c_scale
        kl_c = categorical_kl(responsibilities, pi, kl_c_weight)
        dirichlet_penalty = dirichlet_map_penalty(pi, config.dirichlet_alpha, config.dirichlet_weight)
        usage_penalty = usage_sparsity_penalty(responsibilities, config.usage_sparsity_weight)
        resp_safe = jnp.clip(responsibilities, EPS, 1.0)
        component_entropy = -jnp.mean(jnp.sum(resp_safe * jnp.log(resp_safe), axis=-1))
        pi_safe = jnp.clip(pi, EPS, 1.0)
        pi_entropy = -jnp.sum(pi_safe * jnp.log(pi_safe))
    else:
        rec_loss = reconstruction_loss(batch_x, recon, config.recon_weight, loss_type=config.reconstruction_loss)
        kl_z = kl_divergence(z_mean, z_log, config.kl_weight)
        zero = jnp.array(0.0, dtype=rec_loss.dtype)
        kl_c = zero
        dirichlet_penalty = zero
        usage_penalty = zero
        component_entropy = zero
        pi_entropy = zero

    # Compute classification loss for metrics (unweighted) and for the objective (weighted).
    cls_loss_unweighted = classification_loss(logits, batch_y, 1.0)
    cls_loss_weighted = classification_loss(logits, batch_y, config.label_weight)

    if config.use_contrastive:
        contrastive = _contrastive_loss_stub(z, config.contrastive_weight)
    else:
        contrastive = jnp.array(0.0, dtype=rec_loss.dtype)

    total = rec_loss + kl_z + kl_c + dirichlet_penalty + usage_penalty + cls_loss_weighted + contrastive
    # For readability, expose a variant that excludes global priors/usage terms.
    loss_no_global_priors = rec_loss + kl_z + kl_c + cls_loss_weighted + contrastive
    metrics = {
        "loss": total,
        "loss_no_global_priors": loss_no_global_priors,
        "reconstruction_loss": rec_loss,
        "kl_loss": kl_z + kl_c,
        "kl_z": kl_z,
        "kl_c": kl_c,
        "dirichlet_penalty": dirichlet_penalty,
        "usage_sparsity_loss": usage_penalty,
        "classification_loss": cls_loss_unweighted,
        "weighted_classification_loss": cls_loss_weighted,
        "contrastive_loss": contrastive,
        "component_entropy": component_entropy,
        "pi_entropy": pi_entropy,
    }
    return total, metrics
