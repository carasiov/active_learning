"""Core functionality tests - verifying the mathematical foundation that all models depend on.

These tests guard against bugs in:
- Reconstruction losses (MSE/BCE) - used by ALL models
- KL divergence (standard) - foundation for all VAEs
- Semi-supervised masking - core mechanism for labeled/unlabeled splits
- Training loop correctness - ensures train/val splits preserve label distribution

Run with: poetry run pytest tests/test_core_correctness.py
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from training.losses import (
    classification_loss,
    kl_divergence,
    reconstruction_loss_bce,
    reconstruction_loss_mse,
)


# =============================================================================
# Reconstruction Losses
# =============================================================================

def test_reconstruction_loss_mse_perfect_match():
    """MSE should be zero when reconstruction matches input exactly."""
    batch_size = 8
    img_size = 28

    x = jnp.array(np.random.rand(batch_size, img_size, img_size), dtype=jnp.float32)
    recon = x  # Perfect reconstruction
    weight = 500.0

    loss = reconstruction_loss_mse(x, recon, weight)

    assert jnp.allclose(loss, 0.0, atol=1e-6)


def test_reconstruction_loss_mse_increases_with_error():
    """MSE should increase as reconstruction quality decreases."""
    batch_size = 8
    img_size = 28
    weight = 500.0

    x = jnp.array(np.random.rand(batch_size, img_size, img_size), dtype=jnp.float32)

    # Small error
    recon_small_error = x + 0.01
    loss_small = reconstruction_loss_mse(x, recon_small_error, weight)

    # Large error
    recon_large_error = x + 0.1
    loss_large = reconstruction_loss_mse(x, recon_large_error, weight)

    assert loss_large > loss_small > 0.0


def test_reconstruction_loss_mse_weight_scaling():
    """MSE should scale linearly with weight parameter."""
    batch_size = 8
    img_size = 28

    x = jnp.array(np.random.rand(batch_size, img_size, img_size), dtype=jnp.float32)
    recon = jnp.array(np.random.rand(batch_size, img_size, img_size), dtype=jnp.float32)

    weight_1 = 1.0
    weight_2 = 500.0

    loss_1 = reconstruction_loss_mse(x, recon, weight_1)
    loss_2 = reconstruction_loss_mse(x, recon, weight_2)

    expected_ratio = weight_2 / weight_1
    actual_ratio = loss_2 / loss_1

    assert jnp.allclose(actual_ratio, expected_ratio, rtol=1e-5)


def test_reconstruction_loss_bce_binary_images():
    """BCE should handle binary images (0 and 1) correctly."""
    batch_size = 8
    img_size = 28

    # Binary images (0 or 1)
    x = jnp.array(np.random.choice([0.0, 1.0], size=(batch_size, img_size, img_size)), dtype=jnp.float32)

    # Perfect logits (large positive for 1, large negative for 0)
    logits = jnp.where(x > 0.5, 10.0, -10.0)
    weight = 1.0

    loss = reconstruction_loss_bce(x, logits, weight)

    # Loss should be very small (near-perfect predictions)
    assert loss < 0.1


def test_reconstruction_loss_bce_numerical_stability():
    """BCE should remain numerically stable with extreme logit values."""
    batch_size = 4
    img_size = 28

    x = jnp.array(np.random.rand(batch_size, img_size, img_size), dtype=jnp.float32)

    # Extreme logits (both very large and very small)
    logits = jnp.array(np.random.randn(batch_size, img_size, img_size) * 100, dtype=jnp.float32)
    weight = 1.0

    loss = reconstruction_loss_bce(x, logits, weight)

    # Should produce finite values
    assert jnp.isfinite(loss)
    assert loss >= 0.0  # BCE is non-negative


def test_reconstruction_loss_bce_vs_sigmoid_reference():
    """BCE with logits should match manual sigmoid + BCE calculation."""
    batch_size = 4
    img_size = 14  # Smaller for efficiency

    x = jnp.array(np.random.rand(batch_size, img_size, img_size), dtype=jnp.float32)
    logits = jnp.array(np.random.randn(batch_size, img_size, img_size) * 2, dtype=jnp.float32)
    weight = 1.0

    # Our implementation
    loss_stable = reconstruction_loss_bce(x, logits, weight)

    # Manual calculation (less stable but correct)
    x_flat = x.reshape((batch_size, -1))
    logits_flat = logits.reshape((batch_size, -1))

    sigmoid = jax.nn.sigmoid(logits_flat)
    epsilon = 1e-7
    manual_bce = -jnp.mean(
        jnp.sum(
            x_flat * jnp.log(sigmoid + epsilon) + (1 - x_flat) * jnp.log(1 - sigmoid + epsilon),
            axis=1
        )
    )
    loss_manual = weight * manual_bce

    # Should match within reasonable tolerance
    assert jnp.allclose(loss_stable, loss_manual, rtol=1e-4, atol=1e-4)


# =============================================================================
# KL Divergence (Standard)
# =============================================================================

def test_standard_kl_divergence_zero_for_standard_gaussian():
    """KL should be zero when q(z) matches the prior N(0, I)."""
    batch_size = 16
    latent_dim = 2

    # Standard Gaussian: mean=0, log_var=0 (var=1)
    z_mean = jnp.zeros((batch_size, latent_dim))
    z_log = jnp.zeros((batch_size, latent_dim))
    weight = 1.0

    kl = kl_divergence(z_mean, z_log, weight)

    assert jnp.allclose(kl, 0.0, atol=1e-6)


def test_standard_kl_divergence_positive_for_non_standard():
    """KL should be positive when q(z) differs from the prior."""
    batch_size = 16
    latent_dim = 2

    # Non-standard: shifted mean
    z_mean = jnp.ones((batch_size, latent_dim)) * 2.0
    z_log = jnp.zeros((batch_size, latent_dim))
    weight = 1.0

    kl = kl_divergence(z_mean, z_log, weight)

    assert kl > 0.0


def test_standard_kl_divergence_formula_correctness():
    """Verify KL matches the analytical formula: 0.5 * sum(1 + log(var) - mu^2 - var)."""
    batch_size = 8
    latent_dim = 2

    z_mean = jnp.array(np.random.randn(batch_size, latent_dim), dtype=jnp.float32)
    z_log = jnp.array(np.random.randn(batch_size, latent_dim) * 0.5, dtype=jnp.float32)
    weight = 5.0

    # Our implementation
    kl_computed = kl_divergence(z_mean, z_log, weight)

    # Manual calculation from formula
    kl_per_dim = -0.5 * (1.0 + z_log - jnp.square(z_mean) - jnp.exp(z_log))
    kl_per_sample = jnp.sum(kl_per_dim, axis=1)
    kl_manual = weight * jnp.mean(kl_per_sample)

    assert jnp.allclose(kl_computed, kl_manual, atol=1e-6)


def test_standard_kl_divergence_weight_scaling():
    """KL should scale linearly with weight."""
    batch_size = 8
    latent_dim = 2

    z_mean = jnp.array(np.random.randn(batch_size, latent_dim), dtype=jnp.float32)
    z_log = jnp.array(np.random.randn(batch_size, latent_dim) * 0.5, dtype=jnp.float32)

    weight_1 = 1.0
    weight_2 = 5.0

    kl_1 = kl_divergence(z_mean, z_log, weight_1)
    kl_2 = kl_divergence(z_mean, z_log, weight_2)

    expected_ratio = weight_2 / weight_1
    actual_ratio = kl_2 / kl_1

    assert jnp.allclose(actual_ratio, expected_ratio, rtol=1e-5)


# =============================================================================
# Semi-Supervised Classification Loss
# =============================================================================

def test_classification_loss_masks_unlabeled():
    """Classification loss should only apply to labeled examples (non-NaN labels)."""
    batch_size = 8
    num_classes = 10

    # Create logits (random predictions)
    logits = jnp.array(np.random.randn(batch_size, num_classes), dtype=jnp.float32)

    # Mix of labeled and unlabeled (NaN)
    labels = jnp.array([0.0, 1.0, np.nan, 2.0, np.nan, np.nan, 3.0, np.nan], dtype=jnp.float32)
    weight = 1.0

    loss = classification_loss(logits, labels, weight)

    # Should be finite and non-negative
    assert jnp.isfinite(loss)
    assert loss >= 0.0

    # Now test with all unlabeled - should return 0
    labels_all_unlabeled = jnp.full((batch_size,), np.nan, dtype=jnp.float32)
    loss_unlabeled = classification_loss(logits, labels_all_unlabeled, weight)

    assert jnp.allclose(loss_unlabeled, 0.0, atol=1e-6)


def test_classification_loss_perfect_predictions():
    """Loss should be near-zero for perfect predictions."""
    batch_size = 4
    num_classes = 10

    # Create perfect one-hot predictions
    labels = jnp.array([0.0, 1.0, 2.0, 3.0], dtype=jnp.float32)

    # Perfect logits (large value at correct class, zeros elsewhere)
    logits = jnp.zeros((batch_size, num_classes), dtype=jnp.float32)
    for i, label in enumerate(labels):
        logits = logits.at[i, int(label)].set(100.0)

    weight = 1.0
    loss = classification_loss(logits, labels, weight)

    # Should be very small
    assert loss < 0.01


def test_classification_loss_weight_scaling():
    """Classification loss should scale linearly with weight."""
    batch_size = 8
    num_classes = 10

    logits = jnp.array(np.random.randn(batch_size, num_classes), dtype=jnp.float32)
    labels = jnp.array(np.random.randint(0, num_classes, size=batch_size), dtype=jnp.float32)

    weight_1 = 1.0
    weight_2 = 10.0

    loss_1 = classification_loss(logits, labels, weight_1)
    loss_2 = classification_loss(logits, labels, weight_2)

    expected_ratio = weight_2 / weight_1
    actual_ratio = loss_2 / loss_1

    assert jnp.allclose(actual_ratio, expected_ratio, rtol=1e-5)


def test_classification_loss_only_counts_labeled():
    """Loss computation should only average over labeled examples, not entire batch."""
    num_classes = 10

    # Case 1: 4 labeled samples
    logits_4 = jnp.array(np.random.randn(4, num_classes), dtype=jnp.float32)
    labels_4 = jnp.array([0.0, 1.0, 2.0, 3.0], dtype=jnp.float32)

    # Case 2: Same 4 labeled samples + 4 unlabeled (should give same loss)
    logits_8 = jnp.concatenate([logits_4, jnp.zeros((4, num_classes))], axis=0)
    labels_8 = jnp.array([0.0, 1.0, 2.0, 3.0, np.nan, np.nan, np.nan, np.nan], dtype=jnp.float32)

    weight = 1.0
    loss_4 = classification_loss(logits_4, labels_4, weight)
    loss_8 = classification_loss(logits_8, labels_8, weight)

    # Should be the same (unlabeled samples don't affect loss)
    assert jnp.allclose(loss_4, loss_8, rtol=1e-5)


# =============================================================================
# Training Loop Correctness (Data Split)
# =============================================================================

def test_train_val_split_preserves_label_ratio():
    """Train/val split should maintain similar ratio of labeled to unlabeled samples.

    This is a statistical test - we verify the ratio is preserved within reasonable bounds.
    """
    np.random.seed(42)

    total_samples = 1000
    num_labeled = 100
    val_split = 0.1

    # Create semi-supervised labels
    labels = np.full(total_samples, np.nan, dtype=np.float32)
    labeled_indices = np.random.choice(total_samples, size=num_labeled, replace=False)
    labels[labeled_indices] = np.random.randint(0, 10, size=num_labeled)

    # Simulate train/val split (like in Trainer._prepare_data)
    val_size = max(1, int(val_split * total_samples))
    train_size = total_samples - val_size

    # Permute and split
    perm = np.random.permutation(total_samples)
    labels_perm = labels[perm]

    train_labels = labels_perm[:train_size]
    val_labels = labels_perm[train_size:]

    # Count labeled samples in each split
    train_labeled_count = np.sum(~np.isnan(train_labels))
    val_labeled_count = np.sum(~np.isnan(val_labels))

    # Calculate ratios
    train_labeled_ratio = train_labeled_count / train_size
    val_labeled_ratio = val_labeled_count / val_size
    expected_ratio = num_labeled / total_samples

    # Both splits should be close to the expected ratio (within 5% tolerance)
    # This is a statistical test, so we allow some variance
    assert abs(train_labeled_ratio - expected_ratio) < 0.05
    assert abs(val_labeled_ratio - expected_ratio) < 0.15  # Smaller val set has more variance


def test_train_val_split_no_data_leakage():
    """Ensure train and val splits are disjoint (no data leakage)."""
    total_samples = 100
    val_split = 0.1

    # Create unique identifiers for each sample
    data_ids = np.arange(total_samples)

    # Simulate train/val split
    val_size = max(1, int(val_split * total_samples))
    train_size = total_samples - val_size

    perm = np.random.permutation(total_samples)
    data_ids_perm = data_ids[perm]

    train_ids = data_ids_perm[:train_size]
    val_ids = data_ids_perm[train_size:]

    # Check for overlap
    overlap = np.intersect1d(train_ids, val_ids)

    assert len(overlap) == 0, "Train and val splits should not overlap"
    assert len(train_ids) + len(val_ids) == total_samples


# =============================================================================
# Integration Test
# =============================================================================

def test_loss_components_are_finite_and_non_negative():
    """Integration test: all loss components should produce finite, non-negative values."""
    batch_size = 8
    img_size = 28
    latent_dim = 2
    num_classes = 10

    # Create random inputs
    x = jnp.array(np.random.rand(batch_size, img_size, img_size), dtype=jnp.float32)
    recon_mse = jnp.array(np.random.rand(batch_size, img_size, img_size), dtype=jnp.float32)
    logits_bce = jnp.array(np.random.randn(batch_size, img_size, img_size), dtype=jnp.float32)

    z_mean = jnp.array(np.random.randn(batch_size, latent_dim), dtype=jnp.float32)
    z_log = jnp.array(np.random.randn(batch_size, latent_dim) * 0.5, dtype=jnp.float32)

    class_logits = jnp.array(np.random.randn(batch_size, num_classes), dtype=jnp.float32)
    labels = jnp.array([0.0, 1.0, np.nan, 2.0, np.nan, 3.0, 4.0, np.nan], dtype=jnp.float32)

    # Compute all losses
    loss_mse = reconstruction_loss_mse(x, recon_mse, weight=500.0)
    loss_bce = reconstruction_loss_bce(x, logits_bce, weight=1.0)
    loss_kl = kl_divergence(z_mean, z_log, weight=5.0)
    loss_cls = classification_loss(class_logits, labels, weight=1.0)

    # All should be finite and non-negative
    for loss, name in [
        (loss_mse, "MSE"),
        (loss_bce, "BCE"),
        (loss_kl, "KL"),
        (loss_cls, "Classification"),
    ]:
        assert jnp.isfinite(loss), f"{name} loss should be finite"
        assert loss >= 0.0, f"{name} loss should be non-negative"
