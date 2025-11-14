"""Tau-based latent-only classifier for RCM-VAE.

This module implements the responsibility-based classifier that maps components
to labels via soft count statistics. It replaces the separate classifier head
with a latent-only prediction mechanism.

Theory: See docs/theory/mathematical_specification.md Section 5
"""
from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array


class TauClassifier:
    """Latent-only classifier via responsibility-weighted label mapping.

    Implements p(y|x) = Σ_c q(c|x) τ_{c,y} where τ is built from
    soft counts s_{c,y} accumulated during training.

    Key Features:
    - Accumulates soft counts: s_{c,y} += q(c|x) * 1{y=y_i}
    - Normalizes to probability map: τ_{c,y} = s_{c,y} / Σ_y' s_{c,y'}
    - Stop-gradient on τ in loss (gradients flow through q(c|x) only)
    - Natural multimodality: multiple components per label
    - Enables OOD detection and dynamic label addition

    Example:
        >>> tau_clf = TauClassifier(num_components=10, num_classes=10, alpha_0=1.0)
        >>> # During training
        >>> tau_clf.update_counts(responsibilities, labels, labeled_mask)
        >>> loss = tau_clf.supervised_loss(responsibilities, labels, labeled_mask)
        >>> # During inference
        >>> predictions, class_probs = tau_clf.predict(responsibilities)
    """

    def __init__(
        self,
        num_components: int,
        num_classes: int,
        alpha_0: float = 1.0,
    ):
        """Initialize τ-classifier with smoothing prior.

        Args:
            num_components: Number of mixture components (K)
            num_classes: Number of output classes
            alpha_0: Laplace smoothing prior (prevents zero probabilities)
        """
        self.num_components = num_components
        self.num_classes = num_classes
        self.alpha_0 = alpha_0

        # Initialize soft counts to smoothing prior
        # Shape: [K, num_classes]
        self.s_cy = jnp.ones((num_components, num_classes)) * alpha_0

    def update_counts(
        self,
        responsibilities: Array,  # q(c|x): [batch, K]
        labels: Array,  # y: [batch,]
        labeled_mask: Array,  # bool: [batch,]
    ) -> None:
        """Accumulate soft counts from labeled batch.

        Updates s_{c,y} for each component-label pair based on the
        responsibility-weighted indicator function.

        Args:
            responsibilities: Component responsibilities q(c|x) [batch, K]
            labels: True labels [batch,]
            labeled_mask: Boolean mask for labeled samples [batch,]
        """
        # Filter to labeled samples only
        labeled_resp = responsibilities[labeled_mask]  # [n_labeled, K]
        labeled_y = labels[labeled_mask]  # [n_labeled,]

        if len(labeled_y) == 0:
            # No labeled samples in this batch
            return

        # Vectorized count update:
        #   s_{c,y} += Σ_i q(c|x_i) * 1{y_i=y}
        # Use one-hot encoding followed by matrix multiply to aggregate all
        # component-label pairs in a single operation.
        label_one_hot = jax.nn.one_hot(
            labeled_y, self.num_classes, dtype=labeled_resp.dtype
        )  # [n_labeled, num_classes]

        # responsibilities.T @ one_hot gives shape [K, num_classes]
        count_updates = labeled_resp.T @ label_one_hot

        # Update soft counts in one shot
        self.s_cy = self.s_cy + count_updates

    def get_tau(self) -> Array:
        """Compute normalized τ_{c,y} from soft counts.

        Returns τ matrix where each row is a probability distribution over labels.

        Returns:
            tau: Channel→label probability map [K, num_classes]
        """
        # Normalize each component's counts to form probability distribution
        # τ_{c,y} = s_{c,y} / Σ_y' s_{c,y'}
        return self.s_cy / self.s_cy.sum(axis=1, keepdims=True)

    def predict(
        self, responsibilities: Array  # [batch, K]
    ) -> Tuple[Array, Array]:
        """Predict labels from responsibilities.

        Computes p(y|x) = Σ_c q(c|x) * τ_{c,y} for each sample.

        Args:
            responsibilities: Component responsibilities q(c|x) [batch, K]

        Returns:
            predictions: Predicted class indices [batch,]
            class_probs: Posterior class probabilities [batch, num_classes]
        """
        tau = self.get_tau()  # [K, num_classes]

        # p(y|x) = q(c|x) @ τ_{c,y}
        # Matrix multiplication: [batch, K] @ [K, num_classes] = [batch, num_classes]
        class_probs = responsibilities @ tau

        # Predict class with highest probability
        predictions = jnp.argmax(class_probs, axis=-1)

        return predictions, class_probs

    def supervised_loss(
        self,
        responsibilities: Array,  # [batch, K]
        labels: Array,  # [batch,]
        labeled_mask: Array,  # [batch,]
    ) -> Array:
        """Compute supervised loss with stop-grad on τ.

        Loss: -log Σ_c q(c|x) τ_{c,y_true}

        CRITICAL: Uses stop_gradient on τ so gradients flow through
        responsibilities q(c|x) only, not through count statistics.

        Args:
            responsibilities: Component responsibilities [batch, K]
            labels: True labels [batch,]
            labeled_mask: Boolean mask for labeled samples [batch,]

        Returns:
            Negative log-likelihood (scalar)
        """
        # Get τ with stop-gradient
        tau = jax.lax.stop_gradient(self.get_tau())  # [K, num_classes]

        # Gather τ_{c,y} for true labels
        # tau[:, labels] gives [K, batch] - transpose to [batch, K]
        tau_for_labels = tau[:, labels].T  # [batch, K]

        # Compute p(y_true|x) = Σ_c q(c|x) τ_{c,y_true}
        prob_true = jnp.sum(
            responsibilities * tau_for_labels,  # Element-wise multiply [batch, K]
            axis=-1,
        )  # [batch,]

        # Negative log-likelihood (add epsilon for numerical stability)
        nll = -jnp.log(prob_true + 1e-8)

        # Average over labeled samples only
        labeled_nll = jnp.where(labeled_mask, nll, 0.0)
        num_labeled = jnp.maximum(labeled_mask.sum(), 1.0)  # Avoid division by zero

        return labeled_nll.sum() / num_labeled

    def get_certainty(self, responsibilities: Array) -> Array:
        """Compute prediction certainty for each sample.

        Certainty is defined as max_c (r_c * max_y τ_{c,y}), which measures
        how confidently the most active component maps to any label.

        This enables OOD detection: low certainty indicates the sample is
        not well-represented by any labeled component.

        Args:
            responsibilities: Component responsibilities [batch, K]

        Returns:
            certainty: Prediction certainty scores [batch,]
        """
        tau = self.get_tau()  # [K, num_classes]

        # Maximum label confidence per component
        tau_max = jnp.max(tau, axis=1)  # [K,]

        # Certainty = max_c (r_c * max_y τ_{c,y})
        certainty = jnp.max(
            responsibilities * tau_max[None, :],  # [batch, K]
            axis=1,
        )  # [batch,]

        return certainty

    def get_ood_score(self, responsibilities: Array) -> Array:
        """Compute OOD score for each sample.

        OOD score = 1 - certainty, where high values indicate the sample
        is not owned by any labeled component (likely out-of-distribution).

        Args:
            responsibilities: Component responsibilities [batch, K]

        Returns:
            ood_score: OOD detection scores [batch,]
        """
        certainty = self.get_certainty(responsibilities)
        return 1.0 - certainty

    def get_free_channels(
        self,
        usage_threshold: float = 1e-3,
        confidence_threshold: float = 0.05,
    ) -> Array:
        """Identify free channels available for new labels.

        A channel is considered free if:
        - It has low total usage: Σ_y s_{c,y} < usage_threshold, OR
        - It has low label confidence: max_y τ_{c,y} < confidence_threshold

        Args:
            usage_threshold: Minimum total count to be considered active
            confidence_threshold: Minimum confidence to be considered committed

        Returns:
            free_channel_indices: Indices of free channels [num_free,]
        """
        # Total usage per component
        usage = self.s_cy.sum(axis=1)  # [K,]

        # Maximum confidence per component
        tau = self.get_tau()
        confidence = jnp.max(tau, axis=1)  # [K,]

        # Channel is free if it has low usage OR low confidence
        is_free = (usage < usage_threshold) | (confidence < confidence_threshold)

        # Return indices of free channels
        return jnp.where(is_free, size=self.num_components, fill_value=-1)[0]

    def reset_counts(self) -> None:
        """Reset soft counts to initial state (smoothing prior only).

        Useful for starting fresh accumulation or implementing EMA-based updates.
        """
        self.s_cy = jnp.ones((self.num_components, self.num_classes)) * self.alpha_0

    def get_diagnostics(self) -> dict:
        """Get diagnostic information about τ matrix and component-label associations.

        Returns:
            Dictionary with diagnostic metrics:
                - tau: Full τ matrix [K, num_classes]
                - s_cy: Soft counts [K, num_classes]
                - component_label_confidence: max_y τ_{c,y} per component [K,]
                - component_dominant_label: argmax_y τ_{c,y} per component [K,]
                - components_per_label: Number of components per label [num_classes,]
                - tau_entropy: Entropy of each component's distribution [K,]
        """
        tau = self.get_tau()

        # Component-level metrics
        component_label_confidence = jnp.max(tau, axis=1)  # [K,]
        component_dominant_label = jnp.argmax(tau, axis=1)  # [K,]

        # Label-level metrics (how many components per label)
        components_per_label = jnp.zeros(self.num_classes)
        for y in range(self.num_classes):
            # Count components where label y is dominant
            components_per_label = components_per_label.at[y].add(
                jnp.sum(component_dominant_label == y)
            )

        # Entropy of each component's label distribution
        eps = 1e-8
        tau_safe = jnp.clip(tau, eps, 1.0)
        tau_entropy = -jnp.sum(tau_safe * jnp.log(tau_safe), axis=1)  # [K,]

        return {
            "tau": tau,
            "s_cy": self.s_cy,
            "component_label_confidence": component_label_confidence,
            "component_dominant_label": component_dominant_label,
            "components_per_label": components_per_label,
            "tau_entropy": tau_entropy,
        }
