"""
Test that current SSVAE produces identical results to legacy implementation.

This ensures backward compatibility is maintained.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ssvae import SSVAE, SSVAEConfig

# Import legacy version for comparison
from ssvae.models_legacy import SSVAE as SSVAELegacy


@pytest.fixture
def synthetic_data():
    """Create small synthetic dataset."""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 28, 28).astype(np.float32)
    X = (X - X.min()) / (X.max() - X.min())

    y = np.full(n_samples, np.nan)
    y[:10] = np.random.randint(0, 10, 10)

    return X, y


class TestRefactoredEquivalence:
    """Verify refactored version matches original behavior."""

    def test_both_versions_train_successfully(self, synthetic_data):
        """Both versions should train without errors."""
        X, y = synthetic_data

        config = SSVAEConfig(
            latent_dim=2,
            hidden_dims=(32, 16),
            batch_size=16,
            max_epochs=3,
            random_seed=42,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Original
            path_orig = str(Path(tmpdir) / "original.ckpt")
            vae_orig = SSVAELegacy(input_dim=(28, 28), config=config)
            history_orig = vae_orig.fit(X, y, path_orig)

            # Refactored
            path_refac = str(Path(tmpdir) / "refactored.ckpt")
            vae_refac = SSVAE(input_dim=(28, 28), config=config)
            history_refac = vae_refac.fit(X, y, path_refac)

        # Both should complete training
        assert len(history_orig["loss"]) == 3
        assert len(history_refac["loss"]) == 3

        # Both should decrease loss
        assert history_orig["loss"][-1] < history_orig["loss"][0]
        assert history_refac["loss"][-1] < history_refac["loss"][0]

    def test_same_config_gives_same_initialization(self, synthetic_data):
        """Same seed should give identical initialization."""
        X, y = synthetic_data
        X_test = X[:5]

        config = SSVAEConfig(latent_dim=2, random_seed=42)

        # Create both models with same seed
        vae_orig = SSVAELegacy(input_dim=(28, 28), config=config)
        vae_refac = SSVAE(input_dim=(28, 28), config=config)

        # Get predictions before any training (random init should be identical)
        z_orig, _, _, _ = vae_orig.predict(X_test)
        z_refac, _, _, _ = vae_refac.predict(X_test)

        # Should produce identical latent codes (same random init)
        np.testing.assert_allclose(z_orig, z_refac, rtol=1e-5)

    def test_checkpoint_compatibility(self, synthetic_data):
        """Checkpoints should be loadable by either version."""
        X, y = synthetic_data
        X_test = X[:5]

        config = SSVAEConfig(latent_dim=2, max_epochs=2, batch_size=16, random_seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = str(Path(tmpdir) / "model.ckpt")

            # Train with refactored version
            vae_refac_train = SSVAE(input_dim=(28, 28), config=config)
            vae_refac_train.fit(X, y, weights_path)

            # Load with original version
            vae_orig_load = SSVAELegacy(input_dim=(28, 28), config=config)
            vae_orig_load.load_model_weights(weights_path)

            # Load with refactored version
            vae_refac_load = SSVAE(input_dim=(28, 28), config=config)
            vae_refac_load.load_model_weights(weights_path)

            # Predictions should match
            z_orig, _, _, _ = vae_orig_load.predict(X_test)
            z_refac, _, _, _ = vae_refac_load.predict(X_test)

        np.testing.assert_allclose(z_orig, z_refac, rtol=1e-5)

    def test_refactored_has_same_public_api(self):
        """Refactored version must have same public methods."""
        config = SSVAEConfig()

        vae_orig = SSVAELegacy(input_dim=(28, 28), config=config)
        vae_refac = SSVAE(input_dim=(28, 28), config=config)

        # Check all public methods exist
        public_methods = ["fit", "predict", "load_model_weights"]
        for method in public_methods:
            assert hasattr(vae_orig, method)
            assert hasattr(vae_refac, method)
            assert callable(getattr(vae_orig, method))
            assert callable(getattr(vae_refac, method))

    def test_mixture_prior_works_in_refactored(self, synthetic_data):
        """Mixture prior should work in refactored version."""
        X, y = synthetic_data

        config = SSVAEConfig(
            latent_dim=2,
            prior_type="mixture",
            num_components=5,
            batch_size=16,
            max_epochs=2,
            random_seed=42,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = str(Path(tmpdir) / "mixture.ckpt")
            vae_refac = SSVAE(input_dim=(28, 28), config=config)
            history = vae_refac.fit(X, y, weights_path)

        # Should complete training
        assert len(history["loss"]) == 2
        assert history["kl_c"][-1] >= 0  # Mixture-specific metric


class TestRefactoredComponentsIndependently:
    """Test new components independently."""

    def test_factory_creates_valid_components(self):
        """SSVAEFactory should create valid components."""
        from ssvae.factory import SSVAEFactory

        factory = SSVAEFactory()
        network, state, train_fn, eval_fn, shuffle_rng, prior = factory.create_model(
            input_dim=(28, 28),
            config=SSVAEConfig(latent_dim=2),
        )

        # Check components are valid
        assert network is not None
        assert state is not None
        assert callable(train_fn)
        assert callable(eval_fn)
        assert shuffle_rng is not None
        assert prior is not None
        assert prior.get_prior_type() == "standard"  # Default

    def test_checkpoint_manager_save_load(self, synthetic_data):
        """CheckpointManager should save and load state."""
        from ssvae.checkpoint import CheckpointManager
        from ssvae.factory import SSVAEFactory

        X, y = synthetic_data
        config = SSVAEConfig(latent_dim=2)

        factory = SSVAEFactory()
        _, state1, _, _, _, _ = factory.create_model((28, 28), config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.ckpt")

            # Save
            mgr = CheckpointManager()
            mgr.save(state1, path)

            # Load
            _, state2, _, _, _, _ = factory.create_model((28, 28), config)
            state_loaded = mgr.load(state2, path)

            # Step should match
            assert state_loaded.step == state1.step

    def test_diagnostics_collector_generates_files(self, synthetic_data):
        """DiagnosticsCollector should save mixture stats."""
        from ssvae.diagnostics import DiagnosticsCollector

        X, y = synthetic_data

        config = SSVAEConfig(
            latent_dim=2,
            prior_type="mixture",
            num_components=5,
        )

        vae = SSVAE(input_dim=(28, 28), config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "diagnostics"

            # Collect diagnostics
            collector = DiagnosticsCollector(config)
            collector.collect_mixture_stats(
                apply_fn=vae._apply_fn,
                params=vae.state.params,
                data=X,
                labels=y,
                output_dir=output_dir,
            )

            # Check files were created
            assert (output_dir / "component_usage.npy").exists()
            assert (output_dir / "component_entropy.npy").exists()
            assert (output_dir / "pi.npy").exists()

            # Load and verify
            usage = collector.load_component_usage(output_dir)
            assert usage is not None
            assert usage.shape == (5,)  # 5 components
            assert np.all(usage >= 0)
            assert np.abs(usage.sum() - 1.0) < 0.01  # should sum to ~1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
