"""Unit tests for reconstruction loss functions."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from training.losses import (
    reconstruction_loss,
    reconstruction_loss_mse,
    reconstruction_loss_bce,
)


def test_bce_numerically_stable():
    """Verify BCE handles extreme logits without overflow/underflow."""
    batch_size = 4
    pixels = 28 * 28
    
    # Extreme logits that would break naive sigmoid
    logits = jnp.array([
        [100.0] * pixels,   # Very confident "1"
        [-100.0] * pixels,  # Very confident "0"
        [0.0] * pixels,     # Neutral
        [50.0, -50.0] * (pixels // 2),  # Mixed
    ])
    
    x = jnp.ones((batch_size, pixels)) * 0.5  # All pixels at 0.5
    
    loss = reconstruction_loss_bce(x, logits, weight=1.0)
    
    # Should be finite (no inf/nan)
    assert jnp.isfinite(loss)
    
    # Should be positive (BCE is non-negative)
    assert loss > 0.0


def test_bce_matches_manual_calculation():
    """Verify BCE formula matches hand-calculated values."""
    # Simple case: 2x2 image, batch of 1
    x = jnp.array([[0.0, 1.0, 0.5, 0.25]])  # 4 pixels
    logits = jnp.array([[0.0, 2.0, 0.0, -1.0]])  # 4 logits
    
    # Manual calculation
    # pixel 0: max(0,0) - 0*0 + log(1 + exp(-0)) = log(2) ≈ 0.693
    # pixel 1: max(2,0) - 1*2 + log(1 + exp(-2)) = 2 - 2 + 0.127 ≈ 0.127
    # pixel 2: max(0,0) - 0.5*0 + log(1 + exp(-0)) = log(2) ≈ 0.693
    # pixel 3: max(0,-1) - 0.25*(-1) + log(1 + exp(-1)) = 0.25 + 0.313 ≈ 0.563
    # Sum: 0.693 + 0.127 + 0.693 + 0.563 = 2.076
    # Mean: 2.076 (batch size 1)
    
    expected = 2.076  # Approximate
    computed = reconstruction_loss_bce(x, logits, weight=1.0)
    
    assert jnp.allclose(computed, expected, atol=0.01)


def test_bce_perfect_prediction_gives_zero():
    """Verify BCE → 0 when predictions are perfect."""
    # When x=1 and logits→∞, loss→0
    # When x=0 and logits→-∞, loss→0
    
    x = jnp.array([[1.0, 0.0, 1.0, 0.0]])
    logits = jnp.array([[100.0, -100.0, 100.0, -100.0]])  # Very confident
    
    loss = reconstruction_loss_bce(x, logits, weight=1.0)
    
    # Should be very close to zero
    assert loss < 0.01


def test_mse_vs_bce_outputs_differ():
    """Verify MSE and BCE give different results (sanity check)."""
    x = jnp.ones((8, 28, 28)) * 0.5
    recon = jnp.zeros((8, 28, 28))  # Poor reconstruction
    
    mse = reconstruction_loss_mse(x, recon, weight=1.0)
    bce = reconstruction_loss_bce(x, recon, weight=1.0)
    
    # They should be different (and both positive)
    assert mse > 0.0
    assert bce > 0.0
    assert not jnp.allclose(mse, bce)


def test_dispatcher_mse():
    """Verify dispatcher routes to MSE correctly."""
    x = jnp.ones((4, 28, 28)) * 0.5
    recon = jnp.zeros((4, 28, 28))
    
    loss_direct = reconstruction_loss_mse(x, recon, weight=1.0)
    loss_dispatch = reconstruction_loss(x, recon, weight=1.0, loss_type="mse")
    
    assert jnp.allclose(loss_direct, loss_dispatch)


def test_dispatcher_bce():
    """Verify dispatcher routes to BCE correctly."""
    x = jnp.ones((4, 28, 28)) * 0.5
    logits = jnp.zeros((4, 28, 28))
    
    loss_direct = reconstruction_loss_bce(x, logits, weight=1.0)
    loss_dispatch = reconstruction_loss(x, logits, weight=1.0, loss_type="bce")
    
    assert jnp.allclose(loss_direct, loss_dispatch)


def test_dispatcher_invalid_type():
    """Verify dispatcher raises error for unknown loss type."""
    x = jnp.ones((4, 28, 28))
    recon = jnp.zeros((4, 28, 28))
    
    with pytest.raises(ValueError, match="Unknown reconstruction_loss type"):
        reconstruction_loss(x, recon, weight=1.0, loss_type="invalid")


def test_bce_weight_scaling():
    """Verify weight parameter scales loss correctly."""
    x = jnp.ones((4, 784)) * 0.5
    logits = jnp.zeros((4, 784))
    
    loss_w1 = reconstruction_loss_bce(x, logits, weight=1.0)
    loss_w2 = reconstruction_loss_bce(x, logits, weight=2.0)
    loss_w5 = reconstruction_loss_bce(x, logits, weight=0.5)
    
    assert jnp.allclose(loss_w2, loss_w1 * 2.0)
    assert jnp.allclose(loss_w5, loss_w1 * 0.5)


def test_bce_batch_dimension_invariance():
    """Verify loss averages correctly over batch."""
    # Single sample
    x1 = jnp.ones((1, 784)) * 0.5
    logits1 = jnp.zeros((1, 784))
    loss1 = reconstruction_loss_bce(x1, logits1, weight=1.0)
    
    # Same sample repeated 10 times
    x10 = jnp.tile(x1, (10, 1))
    logits10 = jnp.tile(logits1, (10, 1))
    loss10 = reconstruction_loss_bce(x10, logits10, weight=1.0)
    
    # Should give same loss (averaging over batch)
    assert jnp.allclose(loss1, loss10, atol=1e-5)


def test_bce_gradient_exists():
    """Verify BCE loss is differentiable."""
    x = jnp.ones((4, 784)) * 0.5
    logits = jnp.zeros((4, 784))
    
    def loss_fn(logits):
        return reconstruction_loss_bce(x, logits, weight=1.0)
    
    import jax
    grad = jax.grad(loss_fn)(logits)
    
    # Gradient should exist and be finite
    assert grad.shape == logits.shape
    assert jnp.all(jnp.isfinite(grad))


def test_mse_weight_scaling():
    """Verify MSE weight parameter scales loss correctly."""
    x = jnp.ones((4, 784)) * 0.5
    recon = jnp.zeros((4, 784))
    
    loss_w1 = reconstruction_loss_mse(x, recon, weight=1.0)
    loss_w2 = reconstruction_loss_mse(x, recon, weight=2.0)
    loss_w5 = reconstruction_loss_mse(x, recon, weight=0.5)
    
    assert jnp.allclose(loss_w2, loss_w1 * 2.0)
    assert jnp.allclose(loss_w5, loss_w1 * 0.5)


def test_bce_2d_vs_3d_input():
    """Verify BCE handles both 2D and 3D input shapes."""
    # 2D shape (batch, pixels)
    x_2d = jnp.ones((4, 784)) * 0.5
    logits_2d = jnp.zeros((4, 784))
    loss_2d = reconstruction_loss_bce(x_2d, logits_2d, weight=1.0)
    
    # 3D shape (batch, h, w)
    x_3d = x_2d.reshape((4, 28, 28))
    logits_3d = logits_2d.reshape((4, 28, 28))
    loss_3d = reconstruction_loss_bce(x_3d, logits_3d, weight=1.0)
    
    # Should give same loss
    assert jnp.allclose(loss_2d, loss_3d, atol=1e-5)


def test_mse_gradient_exists():
    """Verify MSE loss is differentiable."""
    x = jnp.ones((4, 784)) * 0.5
    recon = jnp.zeros((4, 784))
    
    def loss_fn(recon):
        return reconstruction_loss_mse(x, recon, weight=1.0)
    
    import jax
    grad = jax.grad(loss_fn)(recon)
    
    # Gradient should exist and be finite
    assert grad.shape == recon.shape
    assert jnp.all(jnp.isfinite(grad))
