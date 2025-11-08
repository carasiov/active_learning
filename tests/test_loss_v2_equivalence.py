"""
Test that losses_v2 (with PriorMode) produces identical results to original losses.py.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from ssvae.config import SSVAEConfig
from ssvae.models import ForwardOutput
from ssvae.priors import MixtureGaussianPrior, StandardGaussianPrior
from training.losses import compute_loss_and_metrics
from training.losses_v2 import compute_loss_and_metrics_v2


@pytest.fixture
def mock_forward_standard():
    """Create mock forward output for standard prior."""
    import jax.random as random

    batch_size = 10
    latent_dim = 2

    def forward_fn(params, batch_x, training, key):
        return ForwardOutput(
            component_logits=None,
            z_mean=jnp.zeros((batch_size, latent_dim)),
            z_log=jnp.zeros((batch_size, latent_dim)),
            z=random.normal(random.PRNGKey(0), (batch_size, latent_dim)),
            recon=batch_x + random.normal(random.PRNGKey(1), batch_x.shape) * 0.1,
            class_logits=jnp.zeros((batch_size, 10)),
            extras={},
        )

    return forward_fn


@pytest.fixture
def mock_forward_mixture():
    """Create mock forward output for mixture prior."""
    import jax.random as random

    batch_size = 10
    num_components = 5
    latent_dim = 2

    def forward_fn(params, batch_x, training, key):
        responsibilities = jnp.ones((batch_size, num_components)) / num_components
        pi = jnp.ones(num_components) / num_components
        recon_per_component = jnp.stack(
            [batch_x + random.normal(random.PRNGKey(i), batch_x.shape) * 0.1
             for i in range(num_components)],
            axis=1,
        )
        expected_recon = jnp.sum(responsibilities[:, :, None, None] * recon_per_component, axis=1)

        return ForwardOutput(
            component_logits=jnp.zeros((batch_size, num_components)),
            z_mean=jnp.zeros((batch_size, latent_dim)),
            z_log=jnp.zeros((batch_size, latent_dim)),
            z=random.normal(random.PRNGKey(0), (batch_size, latent_dim)),
            recon=expected_recon,
            class_logits=jnp.zeros((batch_size, 10)),
            extras={
                "responsibilities": responsibilities,
                "pi": pi,
                "recon_per_component": recon_per_component,
                "pi_logits": jnp.zeros(num_components),
                "component_embeddings": jnp.zeros((num_components, latent_dim)),
            },
        )

    return forward_fn


class TestLossV2StandardPrior:
    """Test losses_v2 with standard prior."""

    def test_standard_prior_identical_to_original(self, mock_forward_standard):
        """Standard prior should produce identical loss to original."""
        import jax

        batch_size = 10
        batch_x = jnp.ones((batch_size, 28, 28))
        batch_y = jnp.full(batch_size, np.nan)  # All unlabeled

        config = SSVAEConfig(
            reconstruction_loss="mse",
            recon_weight=500.0,
            kl_weight=1.0,
            label_weight=0.0,
        )

        params = {}
        rng = jax.random.PRNGKey(42)

        # Original loss computation
        loss_orig, metrics_orig = compute_loss_and_metrics(
            params,
            batch_x,
            batch_y,
            mock_forward_standard,
            config,
            rng,
            training=True,
            kl_c_scale=1.0,
        )

        # New loss computation with prior
        prior = StandardGaussianPrior()
        loss_v2, metrics_v2 = compute_loss_and_metrics_v2(
            params,
            batch_x,
            batch_y,
            mock_forward_standard,
            config,
            prior,
            rng,
            training=True,
            kl_c_scale=1.0,
        )

        # Losses should match
        np.testing.assert_allclose(loss_orig, loss_v2, rtol=1e-5)

        # Key metrics should match
        np.testing.assert_allclose(
            metrics_orig["reconstruction_loss"],
            metrics_v2["reconstruction_loss"],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            metrics_orig["kl_z"],
            metrics_v2["kl_z"],
            rtol=1e-5,
        )


class TestLossV2MixturePrior:
    """Test losses_v2 with mixture prior."""

    def test_mixture_prior_identical_to_original(self, mock_forward_mixture):
        """Mixture prior should produce identical loss to original."""
        import jax

        batch_size = 10
        batch_x = jnp.ones((batch_size, 28, 28))
        batch_y = jnp.full(batch_size, np.nan)

        config = SSVAEConfig(
            prior_type="mixture",
            num_components=5,
            reconstruction_loss="mse",
            recon_weight=500.0,
            kl_weight=1.0,
            kl_c_weight=0.5,
            usage_sparsity_weight=0.1,
            dirichlet_alpha=0.5,
            dirichlet_weight=1.0,
        )

        params = {}
        rng = jax.random.PRNGKey(42)

        # Original loss computation
        loss_orig, metrics_orig = compute_loss_and_metrics(
            params,
            batch_x,
            batch_y,
            mock_forward_mixture,
            config,
            rng,
            training=True,
            kl_c_scale=1.0,
        )

        # New loss computation with prior
        prior = MixtureGaussianPrior()
        loss_v2, metrics_v2 = compute_loss_and_metrics_v2(
            params,
            batch_x,
            batch_y,
            mock_forward_mixture,
            config,
            prior,
            rng,
            training=True,
            kl_c_scale=1.0,
        )

        # Losses should match
        np.testing.assert_allclose(loss_orig, loss_v2, rtol=1e-5)

        # Key metrics should match
        for key in ["reconstruction_loss", "kl_z", "kl_c", "dirichlet_penalty", "usage_sparsity_loss"]:
            if key in metrics_orig and key.replace("_loss", "") in metrics_v2:
                key_v2 = key.replace("_loss", "")  # v2 uses shorter names
                np.testing.assert_allclose(
                    metrics_orig[key],
                    metrics_v2[key_v2],
                    rtol=1e-5,
                    err_msg=f"Mismatch in {key}",
                )

    def test_mixture_has_all_expected_metrics(self, mock_forward_mixture):
        """Mixture prior should return all expected metrics."""
        import jax

        batch_size = 10
        batch_x = jnp.ones((batch_size, 28, 28))
        batch_y = jnp.full(batch_size, np.nan)

        config = SSVAEConfig(
            prior_type="mixture",
            num_components=5,
            usage_sparsity_weight=0.1,
        )

        prior = MixtureGaussianPrior()
        _, metrics = compute_loss_and_metrics_v2(
            {},
            batch_x,
            batch_y,
            mock_forward_mixture,
            config,
            prior,
            jax.random.PRNGKey(42),
            training=True,
        )

        # Should have all mixture-specific metrics
        expected_keys = {
            "loss",
            "reconstruction_loss",
            "classification_loss",
            "kl_z",
            "kl_c",
            "usage_sparsity_loss",  # Backward-compatible naming for Trainer
            "component_entropy",
            "pi_entropy",
        }

        assert expected_keys.issubset(set(metrics.keys()))


class TestPriorSwitching:
    """Test that we can easily switch between priors."""

    def test_same_forward_different_priors(self, mock_forward_standard, mock_forward_mixture):
        """Should be able to use different priors with same model."""
        import jax

        batch_x = jnp.ones((10, 28, 28))
        batch_y = jnp.full(10, np.nan)
        rng = jax.random.PRNGKey(42)

        # Standard prior
        config_standard = SSVAEConfig(prior_type="standard")
        prior_standard = StandardGaussianPrior()
        loss_standard, _ = compute_loss_and_metrics_v2(
            {},
            batch_x,
            batch_y,
            mock_forward_standard,
            config_standard,
            prior_standard,
            rng,
            training=True,
        )

        # Mixture prior
        config_mixture = SSVAEConfig(prior_type="mixture", num_components=5)
        prior_mixture = MixtureGaussianPrior()
        loss_mixture, _ = compute_loss_and_metrics_v2(
            {},
            batch_x,
            batch_y,
            mock_forward_mixture,
            config_mixture,
            prior_mixture,
            rng,
            training=True,
        )

        # Both should produce finite losses
        assert jnp.isfinite(loss_standard)
        assert jnp.isfinite(loss_mixture)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
