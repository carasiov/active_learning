from __future__ import annotations

from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from rcmvae.domain.config import SSVAEConfig
from rcmvae.domain.network import _make_weight_decay_mask


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


def heteroscedastic_reconstruction_loss(
    x: jnp.ndarray,
    mean: jnp.ndarray,
    sigma: jnp.ndarray,
    weight: float,
) -> jnp.ndarray:
    """Heteroscedastic reconstruction loss with learned per-image variance.

    Computes the negative log-likelihood under a Gaussian observation model:
        p(x|mean,σ) = N(x; mean, σ²I)
        -log p(x|mean,σ) = (1/2σ²)||x - mean||² + log σ + const

    The loss balances two objectives:
    1. Reconstruction accuracy: ||x - mean||² / (2σ²)
       - Low σ → high precision required → large penalty for errors
       - High σ → low precision allowed → small penalty for errors
    2. Variance regularization: log σ
       - Prevents trivial solution σ → ∞ to minimize first term
       - Encourages model to be confident (low σ) when appropriate

    Args:
        x: Ground truth images [batch, H, W]
        mean: Predicted mean reconstructions [batch, H, W]
        sigma: Predicted per-image standard deviations [batch,]
        weight: Loss scaling factor

    Returns:
        Weighted scalar loss
    """
    # Compute squared error per image
    diff = jnp.square(x - mean)
    if diff.ndim > 1:
        axes = tuple(range(1, diff.ndim))
        se_per_image = jnp.sum(diff, axis=axes)  # [batch,]
    else:
        se_per_image = diff

    # Numerical stability: ensure sigma is bounded away from zero
    sigma_safe = jnp.maximum(sigma, EPS)

    # Negative log-likelihood: NLL = (1/2σ²)||x - mean||² + log σ
    nll = se_per_image / (2 * sigma_safe ** 2) + jnp.log(sigma_safe)

    return weight * jnp.mean(nll)


def weighted_heteroscedastic_reconstruction_loss(
    x: jnp.ndarray,
    mean_components: jnp.ndarray,
    sigma_components: jnp.ndarray,
    responsibilities: jnp.ndarray,
    weight: float,
) -> jnp.ndarray:
    """Expected heteroscedastic reconstruction loss under q(c|x).

    Computes the expectation of the heteroscedastic NLL over mixture components:
        L = E_q(c|x) [ -log p(x|mean_c, σ_c) ]
          = Σ_c q(c|x) [ ||x - mean_c||²/(2σ_c²) + log σ_c ]

    Each component can learn its own variance characteristics, enabling:
    - Components specializing in clean inputs → low σ
    - Components specializing in noisy inputs → high σ
    - Adaptive uncertainty based on which component is responsible

    Args:
        x: Ground truth images [batch, H, W]
        mean_components: Per-component mean reconstructions [batch, K, H, W]
        sigma_components: Per-component standard deviations [batch, K]
        responsibilities: Component probabilities q(c|x) [batch, K]
        weight: Loss scaling factor

    Returns:
        Weighted scalar loss
    """
    # Compute per-component squared errors
    diff = jnp.square(x[:, None, ...] - mean_components)  # [batch, K, H, W]
    axes = tuple(range(2, diff.ndim))
    se_per_component = jnp.sum(diff, axis=axes)  # [batch, K]

    # Numerical stability
    sigma_safe = jnp.maximum(sigma_components, EPS)

    # Compute per-component NLL
    nll_per_component = (
        se_per_component / (2 * sigma_safe ** 2) + jnp.log(sigma_safe)
    )  # [batch, K]

    # Weight by responsibilities
    weighted_nll = jnp.sum(
        responsibilities * nll_per_component,
        axis=1
    )  # [batch,]

    return weight * jnp.mean(weighted_nll)


def kl_divergence(z_mean: jnp.ndarray, z_log: jnp.ndarray, weight: float) -> jnp.ndarray:
    """Return the KL divergence term scaled by ``weight``."""
    kl = -0.5 * (1.0 + z_log - jnp.square(z_mean) - jnp.exp(z_log))
    if kl.ndim == 2:
        per_sample = jnp.sum(kl, axis=1)
    elif kl.ndim == 3:
        per_sample = jnp.sum(kl, axis=(1, 2))
    else:
        axes = tuple(range(1, kl.ndim))
        per_sample = jnp.sum(kl, axis=axes)
    return weight * jnp.mean(per_sample)


def l1_penalty(params: Dict[str, Dict[str, jnp.ndarray]], mask) -> jnp.ndarray:
    """Compute masked L1 penalty over parameters."""
    masked_abs = jtu.tree_map(
        lambda p, m: jnp.sum(jnp.abs(p)) if m else jnp.array(0.0, dtype=p.dtype),
        params,
        mask,
    )
    leaves = jtu.tree_leaves(masked_abs)
    return jnp.sum(jnp.stack(leaves)) if leaves else jnp.array(0.0)


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
    entropy = -jnp.sum(hat_p * jnp.log(jnp.clip(hat_p, EPS, 1.0)))
    return weight * entropy


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


def tau_classification_loss(
    responsibilities: jnp.ndarray,
    tau: jnp.ndarray,
    labels: jnp.ndarray,
    weight: float,
) -> jnp.ndarray:
    """Compute τ-based classification loss with stop-gradient on τ.

    Loss: -log Σ_c q(c|x) τ_{c,y_true}

    CRITICAL: Uses stop_gradient on τ so gradients flow through
    responsibilities q(c|x) only, not through count statistics.

    Args:
        responsibilities: Component responsibilities q(c|x) [batch, K]
        tau: Channel→label probability map [K, num_classes]
        labels: True labels [batch] (NaN for unlabeled)
        weight: Loss weight

    Returns:
        Weighted negative log-likelihood (scalar)
    """
    # Apply stop-gradient to τ
    tau = jax.lax.stop_gradient(tau)

    # Process labels
    labels = labels.reshape((-1,))
    mask = jnp.logical_not(jnp.isnan(labels))
    mask_float = mask.astype(jnp.float32)
    labels_int = jnp.where(mask, labels, 0.0).astype(jnp.int32)

    # Gather τ_{c,y} for true labels: tau[:, labels] gives [K, batch]
    # We need to handle batch indexing carefully
    batch_size = labels_int.shape[0]
    num_components = tau.shape[0]

    # Use advanced indexing to get τ_{c,y_true} for each sample
    # Create indices for gathering
    component_indices = jnp.arange(num_components)[:, None]  # [K, 1]
    component_indices = jnp.broadcast_to(component_indices, (num_components, batch_size))  # [K, batch]
    label_indices = jnp.broadcast_to(labels_int[None, :], (num_components, batch_size))  # [K, batch]

    # Gather τ values: for each component c and sample i, get τ[c, labels[i]]
    tau_for_labels = tau[component_indices, label_indices]  # [K, batch]
    tau_for_labels = tau_for_labels.T  # [batch, K]

    # Compute p(y_true|x) = Σ_c q(c|x) τ_{c,y_true}
    prob_true = jnp.sum(
        responsibilities * tau_for_labels,  # Element-wise multiply [batch, K]
        axis=-1,
    )  # [batch,]

    # Negative log-likelihood (add epsilon for numerical stability)
    nll = -jnp.log(prob_true + EPS)

    # Average over labeled samples only
    masked_nll_sum = jnp.sum(nll * mask_float)
    labeled_count = jnp.sum(mask_float)
    zero = jnp.array(0.0, dtype=nll.dtype)

    mean_nll = jax.lax.cond(
        labeled_count > 0,
        lambda: masked_nll_sum / labeled_count,
        lambda: zero,
    )

    return weight * mean_nll


def compute_loss_and_metrics_v2(
    params: Dict[str, Dict[str, jnp.ndarray]],
    batch_x: jnp.ndarray,
    batch_y: jnp.ndarray,
    model_apply_fn: Callable,
    config: SSVAEConfig,
    prior,  # PriorMode instance
    rng: jax.Array | None,
    *,
    training: bool,
    kl_c_scale: float = 1.0,
    tau: jnp.ndarray | None = None,  # Optional τ matrix for latent-only classification
    gumbel_temperature: float | None = None,  # Optional temperature override
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute loss and metrics using PriorMode abstraction.

    This is the current loss computation function that delegates to priors
    for their specific KL and reconstruction logic.

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
        tau: Optional τ matrix [K, num_classes] for latent-only classification.
             If provided and config.use_tau_classifier=True, uses τ-based loss.
             Otherwise falls back to standard classifier.
        gumbel_temperature: Optional temperature override for Gumbel-Softmax.

    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    from rcmvae.domain.priors.base import EncoderOutput

    # Forward pass
    use_key = rng if training else None
    forward_output = model_apply_fn(
        params,
        batch_x,
        training=training,
        key=use_key,
        gumbel_temperature=gumbel_temperature,
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

    # Classification loss: use τ-based if enabled and available
    use_tau_classifier = (
        config.use_tau_classifier
        and tau is not None
        and encoder_output.extras is not None
        and encoder_output.extras.get("responsibilities") is not None
    )

    if use_tau_classifier:
        # τ-based latent-only classification
        responsibilities = encoder_output.extras.get("responsibilities")
        cls_loss_unweighted = tau_classification_loss(
            responsibilities, tau, batch_y, weight=1.0
        )
        cls_loss_weighted = tau_classification_loss(
            responsibilities, tau, batch_y, weight=config.label_weight
        )
    else:
        # Standard classifier head
        cls_loss_unweighted = _classification_loss_internal(class_logits, batch_y, weight=1.0)
        cls_loss_weighted = _classification_loss_internal(class_logits, batch_y, weight=config.label_weight)

    # Assemble total loss
    total_kl = sum(
        v for k, v in kl_terms.items()
        if k in ("kl_z", "kl_c", "dirichlet_penalty", "component_diversity")
    )
    if config.l1_weight > 0.0:
        l1_mask = _make_weight_decay_mask(params)
        l1_penalty_value = config.l1_weight * l1_penalty(params, l1_mask)
    else:
        l1_penalty_value = jnp.array(0.0, dtype=recon_loss.dtype)

    total = recon_loss + total_kl + cls_loss_weighted + l1_penalty_value

    # Build metrics dictionary
    metrics = {
        "loss": total,
        "reconstruction_loss": recon_loss,
        "classification_loss": cls_loss_unweighted,
        "weighted_classification_loss": cls_loss_weighted,
        "l1_penalty": l1_penalty_value,
    }

    # Add all KL terms from prior
    for key, value in kl_terms.items():
        metrics[key] = value

    # Ensure all expected keys exist (Trainer expects them all)
    zero = jnp.array(0.0, dtype=recon_loss.dtype)
    if "kl_z" not in metrics:
        metrics["kl_z"] = zero
    if "kl_c" not in metrics:
        metrics["kl_c"] = zero
    if "dirichlet_penalty" not in metrics:
        metrics["dirichlet_penalty"] = zero
    if "component_diversity" not in metrics:
        metrics["component_diversity"] = zero
    if "component_entropy" not in metrics:
        metrics["component_entropy"] = zero
    if "pi_entropy" not in metrics:
        metrics["pi_entropy"] = zero
    if "l1_penalty" not in metrics:
        metrics["l1_penalty"] = zero

    # Aggregate kl_loss for backward compatibility
    metrics["kl_loss"] = metrics["kl_z"] + metrics["kl_c"]

    # Loss without global regularizers (for monitoring)
    metrics["loss_no_global_priors"] = (
        recon_loss
        + metrics["kl_z"]
        + metrics["kl_c"]
        + cls_loss_weighted
    )

    # Add contrastive loss placeholder for backward compatibility
    metrics["contrastive_loss"] = zero

    return total, metrics


def _classification_loss_internal(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    weight: float,
) -> jnp.ndarray:
    """Internal classification loss used by compute_loss_and_metrics_v2.

    Note: This is separate from the public classification_loss() function
    to avoid confusion, as this version uses conditional logic.
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


# Backward compatibility alias
compute_loss_and_metrics = compute_loss_and_metrics_v2
