"""
Refactor Safety Tests - Phase 0

These tests establish baseline behavior before refactoring.
They ensure that the refactored architecture produces identical results.

Run with: JAX_PLATFORMS=cpu poetry run pytest tests/test_refactor_safety.py -v
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from rcmvae.application.model_api import SSVAE
from rcmvae.domain.config import SSVAEConfig


@pytest.fixture
def synthetic_data():
    """Create small synthetic dataset for fast tests."""
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 28, 28).astype(np.float32)
    X = (X - X.min()) / (X.max() - X.min())  # normalize to [0, 1]

    # Semi-supervised: only label 20 samples
    y = np.full(n_samples, np.nan)
    y[:20] = np.random.randint(0, 10, 20)

    return X, y


class TestBaselineBehavior:
    """Capture current behavior as baseline for regression testing."""

    def test_standard_prior_trains_and_losses_decrease(self, synthetic_data):
        """Standard prior should train successfully and reduce loss."""
        X, y = synthetic_data

        config = SSVAEConfig(
            latent_dim=2,
            hidden_dims=(64, 32),
            reconstruction_loss="mse",
            recon_weight=500.0,
            kl_weight=1.0,
            batch_size=32,
            max_epochs=5,
            patience=10,
            random_seed=42,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = str(Path(tmpdir) / "test.ckpt")
            vae = SSVAE(input_dim=(28, 28), config=config)
            history = vae.fit(X, y, weights_path)

        # Baseline assertions
        assert len(history["loss"]) == 5
        assert history["loss"][0] > 0  # initial loss should be positive
        assert history["loss"][-1] < history["loss"][0]  # loss should decrease
        assert history["reconstruction_loss"][-1] > 0
        assert history["kl_loss"][-1] > 0

    def test_mixture_prior_trains_and_losses_decrease(self, synthetic_data):
        """Mixture prior should train successfully and reduce loss."""
        X, y = synthetic_data

        config = SSVAEConfig(
            latent_dim=2,
            hidden_dims=(64, 32),
            prior_type="mixture",
            num_components=5,
            reconstruction_loss="mse",
            recon_weight=500.0,
            kl_weight=1.0,
            kl_c_weight=0.5,
            batch_size=32,
            max_epochs=5,
            patience=10,
            random_seed=42,
            use_tau_classifier=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = str(Path(tmpdir) / "test_mixture.ckpt")
            vae = SSVAE(input_dim=(28, 28), config=config)
            history = vae.fit(X, y, weights_path)

        # Baseline assertions
        assert len(history["loss"]) == 5
        assert history["loss"][0] > 0
        assert history["loss"][-1] < history["loss"][0]
        # Mixture-specific metrics
        assert history["kl_c"][-1] >= 0
        assert history["component_entropy"][-1] >= 0
        assert history["pi_entropy"][-1] >= 0

    def test_predict_output_shapes(self, synthetic_data):
        """Verify predict() returns correct shapes."""
        X, y = synthetic_data
        X_test = X[:10]

        config = SSVAEConfig(latent_dim=2, max_epochs=2, batch_size=32)

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = str(Path(tmpdir) / "test.ckpt")
            vae = SSVAE(input_dim=(28, 28), config=config)
            vae.fit(X, y, weights_path)

            z, recon, pred, cert = vae.predict(X_test)

        assert z.shape == (10, 2)  # latent_dim=2
        assert recon.shape == (10, 28, 28)
        assert pred.shape == (10,)
        assert cert.shape == (10,)
        assert np.all((cert >= 0) & (cert <= 1))  # certainty in [0, 1]

    def test_checkpoint_save_load_cycle(self, synthetic_data):
        """Verify checkpoint save/load produces same predictions."""
        X, y = synthetic_data
        X_test = X[:10]

        config = SSVAEConfig(latent_dim=2, max_epochs=3, batch_size=32, random_seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = str(Path(tmpdir) / "test.ckpt")

            # Train
            vae1 = SSVAE(input_dim=(28, 28), config=config)
            vae1.fit(X, y, weights_path)

            # Load same model twice with same random seed
            vae2 = SSVAE(input_dim=(28, 28), config=config)
            vae2.load_model_weights(weights_path)
            z2, recon2, pred2, cert2 = vae2.predict(X_test)

            vae3 = SSVAE(input_dim=(28, 28), config=config)
            vae3.load_model_weights(weights_path)
            z3, recon3, pred3, cert3 = vae3.predict(X_test)

        # Two fresh loads should produce identical results (deterministic inference)
        np.testing.assert_allclose(z2, z3, rtol=1e-5)
        np.testing.assert_allclose(recon2, recon3, rtol=1e-5)
        np.testing.assert_array_equal(pred2, pred3)
        np.testing.assert_allclose(cert2, cert3, rtol=1e-5)


class TestPublicAPIStability:
    """Ensure public API remains stable after refactoring."""

    def test_ssvae_has_required_methods(self):
        """SSVAE must maintain its public API."""
        config = SSVAEConfig()
        vae = SSVAE(input_dim=(28, 28), config=config)

        # Required methods
        assert hasattr(vae, "fit")
        assert hasattr(vae, "predict")
        assert hasattr(vae, "load_model_weights")
        assert callable(vae.fit)
        assert callable(vae.predict)
        assert callable(vae.load_model_weights)

    def test_ssvae_config_has_required_attrs(self):
        """SSVAEConfig must maintain its essential attributes."""
        config = SSVAEConfig()

        # Architecture
        assert hasattr(config, "latent_dim")
        assert hasattr(config, "encoder_type")
        assert hasattr(config, "decoder_type")
        assert hasattr(config, "hidden_dims")

        # Loss
        assert hasattr(config, "reconstruction_loss")
        assert hasattr(config, "recon_weight")
        assert hasattr(config, "kl_weight")
        assert hasattr(config, "label_weight")

        # Training
        assert hasattr(config, "learning_rate")
        assert hasattr(config, "batch_size")
        assert hasattr(config, "max_epochs")
        assert hasattr(config, "patience")

        # Mixture-specific
        assert hasattr(config, "prior_type")
        assert hasattr(config, "num_components")
        assert hasattr(config, "kl_c_weight")

    def test_predict_signature_unchanged(self, synthetic_data):
        """predict() signature and return format must remain stable."""
        X, y = synthetic_data
        config = SSVAEConfig(max_epochs=2, batch_size=32)

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = str(Path(tmpdir) / "test.ckpt")
            vae = SSVAE(input_dim=(28, 28), config=config)
            vae.fit(X, y, weights_path)

            # Standard predict (4-tuple)
            result = vae.predict(X[:5])
            assert len(result) == 4
            z, recon, pred, cert = result
            assert isinstance(z, np.ndarray)
            assert isinstance(recon, np.ndarray)
            assert isinstance(pred, np.ndarray)
            assert isinstance(cert, np.ndarray)

            # Sample mode
            result_sample = vae.predict(X[:5], sample=True, num_samples=3)
            assert len(result_sample) == 4


class TestComponentFactoryContracts:
    """Test that component factory produces valid components."""

    def test_build_encoder_returns_valid_module(self):
        """Encoder factory should return valid Flax modules."""
        from rcmvae.domain.components.factory import build_encoder

        config_dense = SSVAEConfig(encoder_type="dense", latent_dim=8)
        encoder_dense = build_encoder(config_dense, input_hw=(28, 28))
        assert encoder_dense is not None
        assert hasattr(encoder_dense, "__call__")  # Flax module callable

        config_conv = SSVAEConfig(encoder_type="conv", latent_dim=8)
        encoder_conv = build_encoder(config_conv, input_hw=(28, 28))
        assert encoder_conv is not None

        # Mixture encoder
        config_mixture = SSVAEConfig(
            encoder_type="dense",
            prior_type="mixture",
            num_components=5,
            latent_dim=8,
            use_tau_classifier=False,
        )
        encoder_mixture = build_encoder(config_mixture, input_hw=(28, 28))
        assert encoder_mixture is not None

    def test_build_decoder_returns_valid_module(self):
        """Decoder factory should return valid Flax modules."""
        from rcmvae.domain.components.factory import build_decoder

        config_dense = SSVAEConfig(decoder_type="dense", latent_dim=8)
        decoder = build_decoder(config_dense, input_hw=(28, 28))
        assert decoder is not None
        assert hasattr(decoder, "__call__")

        config_conv = SSVAEConfig(decoder_type="conv", latent_dim=8)
        decoder_conv = build_decoder(config_conv, input_hw=(28, 28))
        assert decoder_conv is not None

    def test_mixture_with_conv_returns_mixture_conv_encoder(self):
        """Mixture prior + conv encoder should build the new convolutional mixture module."""
        from rcmvae.domain.components.encoders import MixtureConvEncoder
        from rcmvae.domain.components.factory import build_encoder

        config = SSVAEConfig(
            encoder_type="conv",
            prior_type="mixture",
            num_components=5,
            latent_dim=4,
            use_tau_classifier=False,
        )

        encoder = build_encoder(config, input_hw=(28, 28))
        assert isinstance(encoder, MixtureConvEncoder)


class TestLossFunctionInvariants:
    """Test mathematical properties of loss functions."""

    def test_kl_divergence_nonnegative(self):
        """KL divergence should always be non-negative."""
        from rcmvae.application.services.loss_pipeline import kl_divergence

        # Standard Gaussian (KL should be ~0)
        z_mean = jnp.zeros((10, 2))
        z_log_var = jnp.zeros((10, 2))
        kl = kl_divergence(z_mean, z_log_var, weight=1.0)
        assert kl >= 0.0
        assert kl < 0.1  # should be very close to 0

        # Non-standard (KL should be positive)
        z_mean = jnp.ones((10, 2))
        z_log_var = jnp.zeros((10, 2))
        kl = kl_divergence(z_mean, z_log_var, weight=1.0)
        assert kl > 0.0

    def test_reconstruction_loss_nonnegative(self):
        """Reconstruction loss should always be non-negative."""
        from rcmvae.application.services.loss_pipeline import reconstruction_loss_mse, reconstruction_loss_bce

        x = jnp.array(np.random.rand(10, 28, 28).astype(np.float32))
        recon = jnp.array(np.random.rand(10, 28, 28).astype(np.float32))

        loss_mse = reconstruction_loss_mse(x, recon, weight=1.0)
        assert loss_mse >= 0.0

        # BCE with logits
        logits = jnp.array(np.random.randn(10, 28, 28).astype(np.float32))
        loss_bce = reconstruction_loss_bce(x, logits, weight=1.0)
        assert loss_bce >= 0.0

    def test_categorical_kl_zero_when_equal(self):
        """KL(q||p) = 0 when q == p."""
        from rcmvae.application.services.loss_pipeline import categorical_kl

        # Uniform distributions
        q = jnp.ones((10, 5)) / 5.0
        pi = jnp.ones(5) / 5.0
        kl = categorical_kl(q, pi, weight=1.0)
        assert jnp.abs(kl) < 1e-5  # should be very close to 0

    def test_usage_sparsity_encourages_concentration(self):
        """Usage sparsity penalty should be lower for concentrated distributions."""
        from rcmvae.application.services.loss_pipeline import usage_sparsity_penalty

        # Uniform distribution (high entropy)
        uniform_resp = jnp.ones((100, 10)) / 10.0
        uniform_penalty = usage_sparsity_penalty(uniform_resp, weight=1.0)

        # Concentrated distribution (low entropy)
        concentrated_resp = jnp.zeros((100, 10))
        concentrated_resp = concentrated_resp.at[:, 0].set(1.0)  # all weight on first component
        concentrated_penalty = usage_sparsity_penalty(concentrated_resp, weight=1.0)

        # Concentrated should have lower penalty (since we minimize entropy)
        assert concentrated_penalty < uniform_penalty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
