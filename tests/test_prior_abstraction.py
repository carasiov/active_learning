"""
Tests for PriorMode abstraction.

Validates that the prior protocol works correctly and that
standard and mixture priors behave as expected.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from ssvae.config import SSVAEConfig
from ssvae.priors import (
    EncoderOutput,
    MixtureGaussianPrior,
    StandardGaussianPrior,
    get_prior,
)


class TestPriorRegistry:
    """Test prior factory and registry."""

    def test_get_standard_prior(self):
        """Factory should create StandardGaussianPrior."""
        prior = get_prior("standard")
        assert isinstance(prior, StandardGaussianPrior)
        assert prior.get_prior_type() == "standard"

    def test_get_mixture_prior(self):
        """Factory should create MixtureGaussianPrior."""
        prior = get_prior("mixture")
        assert isinstance(prior, MixtureGaussianPrior)
        assert prior.get_prior_type() == "mixture"

    def test_unknown_prior_raises_error(self):
        """Unknown prior type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown prior type"):
            get_prior("nonexistent")


class TestStandardGaussianPrior:
    """Test StandardGaussianPrior behavior."""

    def test_compute_kl_returns_only_kl_z(self):
        """Standard prior should return only kl_z term."""
        prior = StandardGaussianPrior()
        config = SSVAEConfig(kl_weight=1.0)

        # Create encoder output (standard Gaussian)
        batch_size = 10
        latent_dim = 2
        encoder_output = EncoderOutput(
            z_mean=jnp.zeros((batch_size, latent_dim)),
            z_log_var=jnp.zeros((batch_size, latent_dim)),
            z=jnp.zeros((batch_size, latent_dim)),
            component_logits=None,
            extras=None,
        )

        kl_terms = prior.compute_kl_terms(encoder_output, config)

        # Should have only kl_z
        assert set(kl_terms.keys()) == {"kl_z"}
        assert kl_terms["kl_z"] >= 0.0

    def test_kl_divergence_zero_for_standard_gaussian(self):
        """KL should be ~0 when q(z|x) = N(0,I)."""
        prior = StandardGaussianPrior()
        config = SSVAEConfig(kl_weight=1.0)

        encoder_output = EncoderOutput(
            z_mean=jnp.zeros((10, 2)),
            z_log_var=jnp.zeros((10, 2)),  # log(1) = 0
            z=jnp.zeros((10, 2)),
        )

        kl_terms = prior.compute_kl_terms(encoder_output, config)
        assert kl_terms["kl_z"] < 0.1  # Should be very close to 0

    def test_reconstruction_loss_mse(self):
        """Standard prior should compute MSE reconstruction loss."""
        prior = StandardGaussianPrior()
        config = SSVAEConfig(reconstruction_loss="mse", recon_weight=1.0)

        x_true = jnp.ones((10, 28, 28))
        x_recon = jnp.zeros((10, 28, 28))
        encoder_output = EncoderOutput(
            z_mean=jnp.zeros((10, 2)),
            z_log_var=jnp.zeros((10, 2)),
            z=jnp.zeros((10, 2)),
        )

        loss = prior.compute_reconstruction_loss(x_true, x_recon, encoder_output, config)
        assert loss > 0.0  # Should have positive loss
        assert jnp.isfinite(loss)

    def test_reconstruction_loss_bce(self):
        """Standard prior should compute BCE reconstruction loss."""
        prior = StandardGaussianPrior()
        config = SSVAEConfig(reconstruction_loss="bce", recon_weight=1.0)

        x_true = jnp.ones((10, 28, 28))
        x_logits = jnp.zeros((10, 28, 28))  # BCE uses logits
        encoder_output = EncoderOutput(
            z_mean=jnp.zeros((10, 2)),
            z_log_var=jnp.zeros((10, 2)),
            z=jnp.zeros((10, 2)),
        )

        loss = prior.compute_reconstruction_loss(x_true, x_logits, encoder_output, config)
        assert loss > 0.0
        assert jnp.isfinite(loss)

    def test_does_not_require_component_embeddings(self):
        """Standard prior should not need component embeddings."""
        prior = StandardGaussianPrior()
        assert prior.requires_component_embeddings() is False


class TestMixtureGaussianPrior:
    """Test MixtureGaussianPrior behavior."""

    def test_compute_kl_returns_all_terms(self):
        """Mixture prior should return kl_z, kl_c, and regularizers."""
        prior = MixtureGaussianPrior()
        config = SSVAEConfig(
            kl_weight=1.0,
            kl_c_weight=1.0,
            usage_sparsity_weight=0.1,
            dirichlet_alpha=0.5,
            dirichlet_weight=1.0,
        )

        # Create encoder output with mixture extras
        batch_size = 10
        num_components = 5
        latent_dim = 2

        responsibilities = jnp.ones((batch_size, num_components)) / num_components
        pi = jnp.ones(num_components) / num_components

        encoder_output = EncoderOutput(
            z_mean=jnp.zeros((batch_size, latent_dim)),
            z_log_var=jnp.zeros((batch_size, latent_dim)),
            z=jnp.zeros((batch_size, latent_dim)),
            component_logits=jnp.zeros((batch_size, num_components)),
            extras={
                "responsibilities": responsibilities,
                "pi": pi,
            },
        )

        kl_terms = prior.compute_kl_terms(encoder_output, config)

        # Should have all mixture-specific terms
        expected_keys = {"kl_z", "kl_c", "dirichlet_penalty", "usage_sparsity", "component_entropy", "pi_entropy"}
        assert set(kl_terms.keys()) == expected_keys

        # All should be finite
        for key, value in kl_terms.items():
            assert jnp.isfinite(value), f"{key} is not finite"

    def test_kl_c_zero_when_q_matches_pi(self):
        """KL_c should be ~0 when q(c|x) = π."""
        prior = MixtureGaussianPrior()
        config = SSVAEConfig(kl_c_weight=1.0)

        num_components = 5
        # Uniform distributions (should match)
        responsibilities = jnp.ones((10, num_components)) / num_components
        pi = jnp.ones(num_components) / num_components

        encoder_output = EncoderOutput(
            z_mean=jnp.zeros((10, 2)),
            z_log_var=jnp.zeros((10, 2)),
            z=jnp.zeros((10, 2)),
            component_logits=jnp.zeros((10, num_components)),
            extras={
                "responsibilities": responsibilities,
                "pi": pi,
            },
        )

        kl_terms = prior.compute_kl_terms(encoder_output, config)
        assert jnp.abs(kl_terms["kl_c"]) < 0.01  # Should be very close to 0

    def test_weighted_reconstruction_loss(self):
        """Mixture prior should compute weighted reconstruction."""
        prior = MixtureGaussianPrior()
        config = SSVAEConfig(reconstruction_loss="mse", recon_weight=1.0)

        batch_size = 10
        num_components = 5

        x_true = jnp.ones((batch_size, 28, 28))
        # Per-component reconstructions
        x_recon_components = jnp.zeros((batch_size, num_components, 28, 28))
        responsibilities = jnp.ones((batch_size, num_components)) / num_components

        encoder_output = EncoderOutput(
            z_mean=jnp.zeros((batch_size, 2)),
            z_log_var=jnp.zeros((batch_size, 2)),
            z=jnp.zeros((batch_size, 2)),
            component_logits=jnp.zeros((batch_size, num_components)),
            extras={
                "responsibilities": responsibilities,
                "pi": jnp.ones(num_components) / num_components,
            },
        )

        loss = prior.compute_reconstruction_loss(
            x_true, x_recon_components, encoder_output, config
        )
        assert loss > 0.0
        assert jnp.isfinite(loss)

    def test_requires_component_embeddings(self):
        """Mixture prior should require component embeddings."""
        prior = MixtureGaussianPrior()
        assert prior.requires_component_embeddings() is True

    def test_missing_extras_raises_error(self):
        """Mixture prior should raise error if extras are missing."""
        prior = MixtureGaussianPrior()
        config = SSVAEConfig()

        # Encoder output without extras
        encoder_output = EncoderOutput(
            z_mean=jnp.zeros((10, 2)),
            z_log_var=jnp.zeros((10, 2)),
            z=jnp.zeros((10, 2)),
            extras=None,  # Missing!
        )

        with pytest.raises(ValueError, match="requires extras"):
            prior.compute_kl_terms(encoder_output, config)


class TestPriorPolymorphism:
    """Test that priors work polymorphically."""

    @pytest.mark.parametrize("prior_type", ["standard", "mixture"])
    def test_all_priors_implement_protocol(self, prior_type):
        """All priors should implement the PriorMode protocol."""
        prior = get_prior(prior_type)

        # Should have all required methods
        assert callable(getattr(prior, "compute_kl_terms", None))
        assert callable(getattr(prior, "compute_reconstruction_loss", None))
        assert callable(getattr(prior, "get_prior_type", None))
        assert callable(getattr(prior, "requires_component_embeddings", None))

    def test_can_switch_priors_at_runtime(self):
        """Should be able to switch between priors dynamically."""
        config = SSVAEConfig()

        # Create different priors
        standard = get_prior("standard")
        mixture = get_prior("mixture")

        # Both should work with the same config
        assert standard.get_prior_type() == "standard"
        assert mixture.get_prior_type() == "mixture"

        # Both should have different requirements
        assert not standard.requires_component_embeddings()
        assert mixture.requires_component_embeddings()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
