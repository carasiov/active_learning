"""
τ-based latent-only classifier for RCM-VAE.

This classifier leverages component specialization by using a component→label
association matrix (τ) computed from accumulated soft counts.

Mathematical Specification (See docs/theory/mathematical_specification.md §5):
    Soft counts:    s_{c,y} ← s_{c,y} + q(c|x) · 1{y=y_i}
    Normalize:      τ_{c,y} = (s_{c,y} + α_0) / Σ_y' (s_{c,y'} + α_0)
    Prediction:     p(y|x) = Σ_c q(c|x) · τ_{c,y}
    Loss:           L_sup = -log Σ_c q(c|x) τ_{c,y_true}  [with stop-grad on τ]

Design:
    The τ matrix is stored as a learnable parameter (but with stop-grad in loss).
    It gets updated periodically from accumulated soft counts (outside gradient descent).
    This keeps the model pure and JAX-friendly while maintaining the τ-based logic.
"""
from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn


class TauClassifier(nn.Module):
    """τ-based classifier using component-label association matrix.

    This classifier uses a τ matrix that maps components to label probabilities.
    The τ matrix is learned from soft counts accumulated during training.

    Attributes:
        num_components: Number of mixture components (K)
        num_classes: Number of output classes
        alpha_0: Dirichlet smoothing parameter for τ normalization (default: 1.0)
    """

    num_components: int
    num_classes: int
    alpha_0: float = 1.0

    @nn.compact
    def __call__(
        self,
        responsibilities: jnp.ndarray,
        *,
        training: bool = False,
    ) -> jnp.ndarray:
        """Forward pass returns class logits.

        Args:
            responsibilities: q(c|x) of shape [batch, num_components]
            training: Unused (for API compatibility with Classifier)

        Returns:
            Class logits (log probabilities) [batch, num_classes]
        """
        # τ matrix is a "parameter" but will be overwritten from soft counts
        # Initialize uniformly: each component equally likely for all classes
        tau = self.param(
            'tau',
            nn.initializers.constant(1.0 / self.num_classes),
            (self.num_components, self.num_classes),
        )

        # Stop-grad on τ: it's updated from soft counts, not via gradient descent
        tau_stopgrad = jax.lax.stop_gradient(tau)

        # Compute p(y|x) = responsibilities @ tau
        # [batch, num_components] @ [num_components, num_classes] -> [batch, num_classes]
        probs = jnp.matmul(responsibilities, tau_stopgrad)

        # Return log probabilities as "logits" for compatibility
        return jnp.log(probs + 1e-8)


# Static utility functions for managing soft counts and τ matrix
# These operate outside the network and are called from the training loop

def compute_tau_from_counts(
    soft_counts: jnp.ndarray,
    alpha_0: float = 1.0,
) -> jnp.ndarray:
    """Compute normalized τ matrix from soft counts.

    Args:
        soft_counts: Accumulated counts [num_components, num_classes]
        alpha_0: Dirichlet smoothing parameter

    Returns:
        τ matrix [num_components, num_classes] with each row summing to 1
    """
    # Add smoothing and normalize
    smoothed = soft_counts + alpha_0
    row_sums = jnp.sum(smoothed, axis=1, keepdims=True)
    tau = smoothed / row_sums
    return tau


def accumulate_soft_counts(
    responsibilities: jnp.ndarray,
    labels: jnp.ndarray,
    num_classes: int,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute soft count increments for a batch.

    Args:
        responsibilities: q(c|x) of shape [batch, num_components]
        labels: Integer labels of shape [batch]
        num_classes: Number of classes
        mask: Optional binary mask for labeled samples [batch]

    Returns:
        Count increments [num_components, num_classes]
    """
    batch_size = responsibilities.shape[0]

    # Handle mask
    if mask is None:
        mask = jnp.ones(batch_size, dtype=jnp.float32)
    else:
        mask = mask.astype(jnp.float32)

    # One-hot encode labels: [batch, num_classes]
    labels_int = labels.astype(jnp.int32)
    labels_onehot = jax.nn.one_hot(labels_int, num_classes)

    # Compute increments: [batch, num_components, num_classes]
    # For each sample: q(c|x) · 1{y=y_i} for all c, y
    increments = (
        responsibilities[:, :, None]  # [batch, num_components, 1]
        * labels_onehot[:, None, :]   # [batch, 1, num_classes]
        * mask[:, None, None]          # [batch, 1, 1]
    )

    # Sum over batch: [num_components, num_classes]
    return jnp.sum(increments, axis=0)


def tau_supervised_loss(
    responsibilities: jnp.ndarray,
    labels: jnp.ndarray,
    tau: jnp.ndarray,
    weight: float = 1.0,
) -> jnp.ndarray:
    """Compute τ-based supervised loss with stop-grad on τ.

    Args:
        responsibilities: q(c|x) of shape [batch, num_components]
        labels: Integer labels of shape [batch]
        tau: τ matrix [num_components, num_classes] (will be stop-grad'd)
        weight: Loss weight multiplier

    Returns:
        Scalar loss: -log Σ_c q(c|x) · τ_{c,y_true}

    Note:
        - Uses stop-grad on τ to prevent gradients flowing to τ
        - Only processes labeled samples (non-NaN labels)
        - Returns 0 if no labeled samples in batch
    """
    # Handle labeled vs unlabeled samples
    labels_flat = labels.reshape((-1,))
    mask = jnp.logical_not(jnp.isnan(labels_flat))
    mask_float = mask.astype(jnp.float32)

    # Stop-grad on τ (counts are not trainable via gradient descent)
    tau_stopgrad = jax.lax.stop_gradient(tau)

    # Compute p(y|x) for all classes
    # [batch, num_components] @ [num_components, num_classes] -> [batch, num_classes]
    class_probs = jnp.matmul(responsibilities, tau_stopgrad)

    # Clip for numerical stability
    class_probs = jnp.clip(class_probs, 1e-8, 1.0)

    # Extract probability for true labels
    labels_int = jnp.where(mask, labels_flat, 0.0).astype(jnp.int32)

    # Compute cross-entropy loss
    # optax expects log-probs, so we pass log(class_probs)
    log_probs = jnp.log(class_probs)
    per_example_loss = optax.softmax_cross_entropy_with_integer_labels(
        log_probs,
        labels_int,
    )

    # Mask out unlabeled samples
    masked_loss_sum = jnp.sum(per_example_loss * mask_float)
    labeled_count = jnp.sum(mask_float)

    # Return mean loss over labeled samples (or 0 if no labels)
    zero = jnp.array(0.0, dtype=per_example_loss.dtype)
    mean_loss = jax.lax.cond(
        labeled_count > 0,
        lambda: masked_loss_sum / labeled_count,
        lambda: zero,
    )

    return weight * mean_loss


def predict_from_tau(
    responsibilities: jnp.ndarray,
    tau: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Predict class labels and probabilities using τ matrix.

    Args:
        responsibilities: q(c|x) of shape [batch, num_components]
        tau: τ matrix [num_components, num_classes]

    Returns:
        Tuple of:
            - predictions: Predicted class indices [batch]
            - probabilities: Class probabilities p(y|x) [batch, num_classes]
    """
    # Compute p(y|x) = responsibilities @ tau
    probs = jnp.matmul(responsibilities, tau)

    # Get predicted class
    predictions = jnp.argmax(probs, axis=-1)

    return predictions, probs


def get_certainty(
    responsibilities: jnp.ndarray,
    tau: jnp.ndarray,
) -> jnp.ndarray:
    """Compute prediction certainty for each sample.

    Args:
        responsibilities: q(c|x) of shape [batch, num_components]
        tau: τ matrix [num_components, num_classes]

    Returns:
        Certainty scores [batch] in range [0, 1]

    Formula:
        certainty(x) = max_c (r_c · max_y τ_{c,y})
    """
    # Get maximum τ for each component: [num_components]
    max_tau_per_component = jnp.max(tau, axis=1)

    # Weight by responsibilities and take max
    weighted = responsibilities * max_tau_per_component[None, :]
    certainty = jnp.max(weighted, axis=1)

    return certainty


def get_ood_score(
    responsibilities: jnp.ndarray,
    tau: jnp.ndarray,
) -> jnp.ndarray:
    """Compute OOD score for each sample.

    Args:
        responsibilities: q(c|x) of shape [batch, num_components]
        tau: τ matrix [num_components, num_classes]

    Returns:
        OOD scores [batch] in range [0, 1]
        Higher values indicate more likely to be out-of-distribution.

    Formula (from Mathematical Specification §6):
        s_OOD(x) = 1 - max_c (q(c|x) · max_y τ_{c,y})
    """
    return 1.0 - get_certainty(responsibilities, tau)


def get_free_channels(
    usage: jnp.ndarray,
    tau: jnp.ndarray,
    usage_threshold: float = 1e-3,
    tau_threshold: float = 0.05,
) -> jnp.ndarray:
    """Identify free channels available for new labels.

    Args:
        usage: Empirical component usage p̂(c) of shape [num_components]
        tau: τ matrix [num_components, num_classes]
        usage_threshold: Channels with usage below this are considered free
        tau_threshold: Channels with max_y τ_{c,y} below this are considered free

    Returns:
        Boolean array [num_components] indicating free channels

    Criteria (from Mathematical Specification §7):
        A channel is free if:
            usage(c) < 1e-3  OR  max_y τ_{c,y} < 0.05
    """
    max_tau_per_component = jnp.max(tau, axis=1)

    is_low_usage = usage < usage_threshold
    is_low_confidence = max_tau_per_component < tau_threshold

    is_free = jnp.logical_or(is_low_usage, is_low_confidence)

    return is_free


def update_tau_in_params(
    params: dict,
    soft_counts: jnp.ndarray,
    alpha_0: float = 1.0,
) -> dict:
    """Update τ parameter in model parameters from soft counts.

    This function creates a new params dict with updated τ matrix.
    It should be called periodically during training (e.g., after each epoch)
    to update the classifier's τ matrix from accumulated soft counts.

    Args:
        params: Model parameters dictionary (typically state.params)
        soft_counts: Accumulated soft counts [num_components, num_classes]
        alpha_0: Dirichlet smoothing parameter

    Returns:
        Updated params dictionary with new τ matrix

    Example:
        >>> # In training loop
        >>> soft_counts = jnp.zeros((num_components, num_classes))
        >>> for batch in epoch:
        >>>     # ... forward pass to get responsibilities ...
        >>>     batch_counts = accumulate_soft_counts(resp, labels, num_classes)
        >>>     soft_counts += batch_counts
        >>> # Update τ at end of epoch
        >>> new_tau = compute_tau_from_counts(soft_counts, alpha_0)
        >>> new_params = update_tau_in_params(state.params, soft_counts, alpha_0)
        >>> state = state.replace(params=new_params)
    """
    import copy

    # Compute new τ from counts
    new_tau = compute_tau_from_counts(soft_counts, alpha_0)

    # Create a copy of params (Flax params are frozen dicts, need to unfreeze)
    from flax.core import freeze, unfreeze

    params_copy = unfreeze(params)

    # Update τ in classifier parameters
    # The path is: params['classifier']['tau']
    if 'classifier' in params_copy and 'tau' in params_copy['classifier']:
        params_copy['classifier']['tau'] = new_tau
    else:
        raise ValueError(
            "τ parameter not found in params. "
            "Make sure you're using TauClassifier and it has been initialized."
        )

    # Freeze and return
    return freeze(params_copy)


def extract_tau_from_params(params: dict) -> jnp.ndarray:
    """Extract current τ matrix from model parameters.

    Args:
        params: Model parameters dictionary

    Returns:
        τ matrix [num_components, num_classes]
    """
    if 'classifier' in params and 'tau' in params['classifier']:
        return params['classifier']['tau']
    else:
        raise ValueError(
            "τ parameter not found in params. "
            "Make sure you're using TauClassifier and it has been initialized."
        )
