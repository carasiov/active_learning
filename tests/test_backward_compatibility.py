"""Backward compatibility tests for mixture prior changes."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ssvae import SSVAE, SSVAEConfig


def test_default_config_unchanged():
    """Verify SSVAEConfig() defaults to standard prior."""
    config = SSVAEConfig()
    
    assert config.prior_type == "standard"
    assert config.num_components == 10
    assert config.component_kl_weight == 0.1
    
    # Verify it's backward compatible
    assert config.encoder_type == "dense"
    assert config.decoder_type == "dense"
    assert config.latent_dim == 2


def test_standard_mode_trains_identically():
    """Verify standard prior mode trains without using mixture code paths."""
    config = SSVAEConfig(
        prior_type="standard",
        latent_dim=2,
        hidden_dims=(64, 32),
        max_epochs=2,
        batch_size=32,
    )
    
    X = np.random.rand(100, 28, 28).astype(np.float32)
    y = np.full(100, np.nan)
    y[:10] = np.random.randint(0, 10, 10)
    
    vae = SSVAE(input_dim=(28, 28), config=config)
    history = vae.fit(X, y, weights_path="/tmp/test_standard_compat.ckpt")
    
    # Verify metrics exist
    assert "loss" in history
    assert "kl_loss" in history
    assert "reconstruction_loss" in history
    
    # Verify component_entropy is 0 for standard mode
    assert "component_entropy" in history
    assert all(val == 0.0 for val in history["component_entropy"])


def test_standard_encoder_still_works():
    """Verify DenseEncoder and ConvEncoder work unchanged."""
    # Dense encoder
    config_dense = SSVAEConfig(encoder_type="dense", prior_type="standard")
    vae_dense = SSVAE(input_dim=(28, 28), config=config_dense)
    
    X = np.random.rand(10, 28, 28).astype(np.float32)
    latent, recon, pred_class, pred_certainty = vae_dense.predict(X)
    
    assert latent.shape == (10, 2)
    assert recon.shape == (10, 28, 28)
    
    # Conv encoder
    config_conv = SSVAEConfig(
        encoder_type="conv",
        decoder_type="conv",
        prior_type="standard"
    )
    vae_conv = SSVAE(input_dim=(28, 28), config=config_conv)
    latent_conv, recon_conv, _, _ = vae_conv.predict(X)
    
    assert latent_conv.shape == (10, 2)
    assert recon_conv.shape == (10, 28, 28)


def test_config_serialization_includes_mixture_fields():
    """Verify config.get_informative_hyperparameters includes mixture params."""
    config = SSVAEConfig(
        prior_type="mixture",
        num_components=7,
        component_kl_weight=0.05,
    )
    
    hparams = config.get_informative_hyperparameters()
    
    assert "prior_type" in hparams
    assert "num_components" in hparams
    assert "component_kl_weight" in hparams
    
    assert hparams["prior_type"] == "mixture"
    assert hparams["num_components"] == 7
    assert hparams["component_kl_weight"] == 0.05


def test_predict_handles_both_modes():
    """Verify predict() works for both standard and mixture modes."""
    X_train = np.random.rand(50, 28, 28).astype(np.float32)
    y_train = np.full(50, np.nan)
    X_test = np.random.rand(5, 28, 28).astype(np.float32)
    
    # Standard mode
    vae_std = SSVAE(
        input_dim=(28, 28),
        config=SSVAEConfig(prior_type="standard", max_epochs=1)
    )
    vae_std.fit(X_train, y_train, weights_path="/tmp/test_predict_std.ckpt")
    latent_std, recon_std, _, _ = vae_std.predict(X_test)
    
    # Mixture mode
    vae_mix = SSVAE(
        input_dim=(28, 28),
        config=SSVAEConfig(prior_type="mixture", num_components=3, max_epochs=1)
    )
    vae_mix.fit(X_train, y_train, weights_path="/tmp/test_predict_mix.ckpt")
    latent_mix, recon_mix, _, _ = vae_mix.predict(X_test)
    
    # Both should return same shapes
    assert latent_std.shape == latent_mix.shape == (5, 2)
    assert recon_std.shape == recon_mix.shape == (5, 28, 28)
