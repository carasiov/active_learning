from __future__ import annotations

from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import optax

from ssvae.config import SSVAEConfig

def reconstruction_loss_mse(x: jnp.ndarray, recon: jnp.ndarray, weight: float) -> jnp.ndarray:
    """Mean squared error reconstruction loss.
    
    Args:
        x: Target images, shape (batch, H, W) or (batch, H*W).
        recon: Decoder outputs, same shape as x.
        weight: Scaling factor for the loss.
    
    Returns:
        Weighted MSE loss, scalar.
    
    Notes:
        Appropriate for continuous-valued data. For binary data,
        consider using reconstruction_loss_bce instead.
    """
    diff = jnp.square(x - recon)
    if diff.ndim > 1:
        axes = tuple(range(1, diff.ndim))
        per_sample = jnp.mean(diff, axis=axes)
    else:
        per_sample = diff
    return weight * jnp.mean(per_sample)


def reconstruction_loss_bce(x: jnp.ndarray, logits: jnp.ndarray, weight: float) -> jnp.ndarray:
    """Binary cross-entropy with logits (numerically stable).
    
    Args:
        x: Target images in [0, 1], shape (batch, H, W) or (batch, H*W).
        logits: Raw decoder outputs (pre-sigmoid), same shape as x.
        weight: Scaling factor for the loss.
    
    Returns:
        Weighted BCE loss, scalar.
    
    Notes:
        Uses log-sum-exp trick for numerical stability:
        BCE = max(logits, 0) - x * logits + log(1 + exp(-|logits|))
        
        This formulation avoids computing sigmoid explicitly and handles
        large positive/negative logits gracefully.
    
    References:
        - Kingma & Welling (2013): Auto-Encoding Variational Bayes
        - PyTorch F.binary_cross_entropy_with_logits implementation
    """
    # Flatten spatial dimensions if needed
    if x.ndim > 2:
        x_flat = x.reshape((x.shape[0], -1))
        logits_flat = logits.reshape((logits.shape[0], -1))
    else:
        x_flat = x
        logits_flat = logits
    
    # Numerically stable BCE computation
    max_val = jnp.maximum(logits_flat, 0.0)
    per_pixel_loss = max_val - x_flat * logits_flat + jnp.log1p(jnp.exp(-jnp.abs(logits_flat)))
    
    # Sum over pixels, average over batch
    per_sample_loss = jnp.sum(per_pixel_loss, axis=1)
    batch_loss = jnp.mean(per_sample_loss)
    
    return weight * batch_loss


def reconstruction_loss(
    x: jnp.ndarray,
    recon: jnp.ndarray,
    weight: float,
    loss_type: str = "mse",
) -> jnp.ndarray:
    """Compute reconstruction loss with configurable type.
    
    Args:
        x: Target images.
        recon: Decoder outputs (raw values for MSE, logits for BCE).
        weight: Scaling factor for the loss.
        loss_type: Loss function type ("mse" or "bce").
    
    Returns:
        Weighted reconstruction loss, scalar.
    
    Raises:
        ValueError: If loss_type is not recognized.
    """
    if loss_type == "mse":
        return reconstruction_loss_mse(x, recon, weight)
    elif loss_type == "bce":
        return reconstruction_loss_bce(x, recon, weight)
    else:
        raise ValueError(
            f"Unknown reconstruction_loss type: '{loss_type}'. "
            f"Valid options: 'mse', 'bce'"
        )


def kl_divergence(z_mean: jnp.ndarray, z_log: jnp.ndarray, weight: float) -> jnp.ndarray:
    """Return the KL divergence term scaled by ``weight``."""
    kl = -0.5 * (1.0 + z_log - jnp.square(z_mean) - jnp.exp(z_log))
    return weight * jnp.mean(jnp.sum(kl, axis=1))


def kl_divergence_mixture(
    component_logits: jnp.ndarray,
    z_mean: jnp.ndarray,
    z_log: jnp.ndarray,
    weight: float,
) -> jnp.ndarray:
    """Return mixture KL divergence KL(q(c,z|x) || Uniform(c)*N(z|0,I))."""
    responsibilities = jax.nn.softmax(component_logits, axis=-1)
    num_components = component_logits.shape[-1]
    
    # KL for each component against standard Gaussian
    kl_per_component = -0.5 * (1.0 + z_log - jnp.square(z_mean) - jnp.exp(z_log))
    kl_per_component_sum = jnp.sum(kl_per_component, axis=-1)
    
    # Entropy of component distribution: -sum(r * log(r))
    log_responsibilities = jnp.log(responsibilities + 1e-10)
    component_entropy = -jnp.sum(responsibilities * log_responsibilities, axis=-1)
    
    # KL(q(c) || Uniform) = log(K) - H(q(c))
    kl_component = jnp.log(float(num_components)) - component_entropy
    
    # Total mixture KL: E_q(c)[KL(q(z|c) || p(z))] + KL(q(c) || p(c))
    mixture_kl = kl_per_component_sum + kl_component
    
    return weight * jnp.mean(mixture_kl)


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
    model_apply_fn: Callable[..., Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]],
    config: SSVAEConfig,
    rng: jax.Array | None,
    *,
    training: bool,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Mirror ``SSVAE._loss_and_metrics`` using the pure loss helpers."""
    use_key = rng if training else None
    component_logits, z_mean, z_log, z, recon, logits = model_apply_fn(
        params,
        batch_x,
        training=training,
        key=use_key,
    )

    rec_loss = reconstruction_loss(
        batch_x, 
        recon, 
        config.recon_weight,
        loss_type=config.reconstruction_loss
    )
    
    # Dispatch KL computation based on prior type
    if component_logits is not None:
        kl_loss = kl_divergence_mixture(component_logits, z_mean, z_log, config.component_kl_weight)
        # Compute component entropy metric
        responsibilities = jax.nn.softmax(component_logits, axis=-1)
        log_resp = jnp.log(responsibilities + 1e-10)
        component_entropy = -jnp.mean(jnp.sum(responsibilities * log_resp, axis=-1))
    else:
        kl_loss = kl_divergence(z_mean, z_log, config.kl_weight)
        component_entropy = jnp.array(0.0, dtype=kl_loss.dtype)
    
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
        "component_entropy": component_entropy,
    }
    return total, metrics
