"""Integration tests for mixture prior SSVAE."""
from __future__ import annotations

import numpy as np

from ssvae import SSVAE, SSVAEConfig


def test_mixture_model_trains_without_error():
    """Verify mixture SSVAE trains successfully on small dataset."""
    config = SSVAEConfig(
        prior_type="mixture",
        num_components=3,
        component_kl_weight=0.1,
        latent_dim=2,
        hidden_dims=(64, 32),
        max_epochs=2,
        batch_size=32,
        patience=10,
        use_tau_classifier=False,
    )
    
    # Create small dummy dataset
    num_samples = 100
    X = np.random.rand(num_samples, 28, 28).astype(np.float32)
    y = np.full(num_samples, np.nan)  # All unlabeled
    y[:10] = np.random.randint(0, 10, 10)  # Label first 10
    
    vae = SSVAE(input_dim=(28, 28), config=config)
    
    # Should complete without exception
    history = vae.fit(X, y, weights_path="/tmp/test_mixture_ssvae.ckpt")
    
    assert history is not None
    assert len(history) > 0


def test_mixture_vs_standard_mode_checkpoint_isolation():
    """Verify mixture and standard models save/load configs correctly."""
    X = np.random.rand(50, 28, 28).astype(np.float32)
    y = np.full(50, np.nan)
    
    # Train standard model
    config_std = SSVAEConfig(prior_type="standard", max_epochs=1)
    vae_std = SSVAE(input_dim=(28, 28), config=config_std)
    vae_std.fit(X, y, weights_path="/tmp/test_standard.ckpt")
    
    # Train mixture model
    config_mix = SSVAEConfig(
        prior_type="mixture",
        num_components=3,
        max_epochs=1,
        use_tau_classifier=False,
    )
    vae_mix = SSVAE(input_dim=(28, 28), config=config_mix)
    vae_mix.fit(X, y, weights_path="/tmp/test_mixture.ckpt")
    
    # Verify configs persisted correctly
    assert vae_std.config.prior_type == "standard"
    assert vae_mix.config.prior_type == "mixture"
    assert vae_mix.config.num_components == 3


def test_conv_mixture_initializes_and_fits():
    """Mixture prior should work with convolutional encoders."""
    config = SSVAEConfig(
        encoder_type="conv",
        decoder_type="conv",
        prior_type="mixture",
        num_components=4,
        max_epochs=1,
        batch_size=16,
        use_tau_classifier=False,
    )

    X = np.random.rand(64, 28, 28).astype(np.float32)
    y = np.full(64, np.nan)

    vae = SSVAE(input_dim=(28, 28), config=config)
    history = vae.fit(X, y, weights_path="/tmp/test_conv_mixture.ckpt")

    assert history is not None
    assert len(history) > 0


def test_mixture_predict_output_shapes():
    """Verify predict() returns correct shapes with mixture prior."""
    config = SSVAEConfig(
        prior_type="mixture",
        num_components=5,
        latent_dim=2,
        max_epochs=1,
        use_tau_classifier=False,
    )
    
    X_train = np.random.rand(50, 28, 28).astype(np.float32)
    y_train = np.full(50, np.nan)
    
    vae = SSVAE(input_dim=(28, 28), config=config)
    vae.fit(X_train, y_train, weights_path="/tmp/test_mixture_predict.ckpt")
    
    X_test = np.random.rand(10, 28, 28).astype(np.float32)
    latent, recon, pred_class, pred_certainty = vae.predict(X_test)
    latent_mix, recon_mix, pred_class_mix, pred_certainty_mix, q_c, pi = vae.predict(X_test, return_mixture=True)

    assert latent.shape == (10, 2)
    assert recon.shape == (10, 28, 28)
    assert pred_class.shape == (10,)
    assert pred_certainty.shape == (10,)
    assert q_c.shape == (10, 5)
    assert pi.shape == (5,)
    assert np.allclose(latent, latent_mix)
    assert np.allclose(recon, recon_mix)


def test_mixture_sample_prediction():
    """Verify sample=True works with mixture prior."""
    config = SSVAEConfig(
        prior_type="mixture",
        num_components=3,
        latent_dim=2,
        max_epochs=1,
        use_tau_classifier=False,
    )
    
    X_train = np.random.rand(50, 28, 28).astype(np.float32)
    y_train = np.full(50, np.nan)
    
    vae = SSVAE(input_dim=(28, 28), config=config)
    vae.fit(X_train, y_train, weights_path="/tmp/test_mixture_sample.ckpt")
    
    X_test = np.random.rand(5, 28, 28).astype(np.float32)
    latent, recon, pred_class, pred_certainty, q_c, pi = vae.predict(
        X_test, sample=True, num_samples=10, return_mixture=True
    )
    
    assert latent.shape == (10, 5, 2)
    assert recon.shape == (10, 5, 28, 28)
    assert pred_class.shape == (10, 5)
    assert pred_certainty.shape == (10, 5)
    assert q_c.shape == (10, 5, 3)
    assert pi.shape == (3,)
