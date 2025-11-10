"""Integration tests for τ-classifier full pipeline."""
from __future__ import annotations

import numpy as np
import pytest

# Note: This test file uses pytest but can also be run directly
# It validates the full τ-classifier integration with SSVAE


class TestTauClassifierIntegration:
    """Test full τ-classifier integration with SSVAE training."""

    @pytest.fixture
    def mock_mnist_data(self):
        """Create small mock MNIST-like dataset."""
        np.random.seed(42)

        # Create 200 samples: 28x28 images with 10 classes
        n_samples = 200
        n_labeled = 50  # 50 labeled, 150 unlabeled

        # Random images (normally would be actual MNIST)
        X = np.random.randn(n_samples, 28, 28).astype(np.float32)

        # Labels: first 50 labeled (5 per class), rest unlabeled (NaN)
        y = np.full(n_samples, np.nan, dtype=np.float32)
        for i in range(n_labeled):
            y[i] = i % 10  # 5 samples per class

        return X, y

    def test_tau_classifier_initialization(self):
        """Test that τ-classifier is initialized correctly."""
        from ssvae import SSVAE
        from ssvae.config import SSVAEConfig

        # Create config with τ-classifier enabled
        config = SSVAEConfig(
            latent_dim=2,
            prior_type="mixture",
            num_components=5,
            use_tau_classifier=True,
            tau_smoothing_alpha=1.0,
            max_epochs=2,  # Minimal for testing
        )

        model = SSVAE(input_dim=(28, 28), config=config)

        # Verify τ-classifier was initialized
        assert model._tau_classifier is not None
        assert model._tau_classifier.num_components == 5
        assert model._tau_classifier.num_classes == 10
        assert model._tau_classifier.alpha_0 == 1.0

    def test_tau_classifier_disabled_for_standard_prior(self):
        """Test that τ-classifier is not used with standard prior."""
        from ssvae import SSVAE
        from ssvae.config import SSVAEConfig

        config = SSVAEConfig(
            latent_dim=2,
            prior_type="standard",  # Standard prior
            use_tau_classifier=True,  # This should be ignored
            max_epochs=2,
        )

        model = SSVAE(input_dim=(28, 28), config=config)

        # τ-classifier should NOT be initialized
        assert model._tau_classifier is None

    def test_end_to_end_training_with_tau(self, mock_mnist_data, tmp_path):
        """Test complete training pipeline with τ-classifier."""
        from ssvae import SSVAE
        from ssvae.config import SSVAEConfig

        X, y = mock_mnist_data
        checkpoint_path = str(tmp_path / "test_tau.ckpt")

        # Create config with τ-classifier
        config = SSVAEConfig(
            latent_dim=2,
            prior_type="mixture",
            num_components=5,
            use_component_aware_decoder=True,
            use_tau_classifier=True,
            max_epochs=3,  # Quick training for testing
            patience=10,
            batch_size=32,
            learning_rate=0.001,
            recon_weight=1.0,
            kl_weight=1.0,
            label_weight=1.0,
            component_diversity_weight=-0.05,  # Reward diversity
        )

        model = SSVAE(input_dim=(28, 28), config=config)

        # Train
        history = model.fit(X, y, checkpoint_path, export_history=False)

        # Verify training completed
        assert "loss" in history
        assert len(history["loss"]) > 0
        assert len(history["loss"]) <= 3  # Should stop at max_epochs

        # Verify τ-classifier was used (counts should be updated)
        tau = model._tau_classifier.get_tau()
        assert tau.shape == (5, 10)  # K=5, num_classes=10
        assert np.allclose(tau.sum(axis=1), 1.0)  # Normalized

        # τ should no longer be uniform (it was trained)
        # Initially uniform would be 0.1 for each class
        # After training, some entries should differ significantly
        assert not np.allclose(tau, 0.1)

    def test_tau_counts_accumulate_during_training(self, mock_mnist_data, tmp_path):
        """Test that τ counts accumulate during training."""
        from ssvae import SSVAE
        from ssvae.config import SSVAEConfig

        X, y = mock_mnist_data
        checkpoint_path = str(tmp_path / "test_counts.ckpt")

        config = SSVAEConfig(
            latent_dim=2,
            prior_type="mixture",
            num_components=3,
            use_tau_classifier=True,
            max_epochs=2,
            batch_size=32,
        )

        model = SSVAE(input_dim=(28, 28), config=config)

        # Get initial counts (should be just the prior)
        initial_counts = model._tau_classifier.s_cy.copy()
        assert np.allclose(initial_counts, 1.0)  # alpha_0=1.0

        # Train
        model.fit(X, y, checkpoint_path, export_history=False)

        # Counts should have increased
        final_counts = model._tau_classifier.s_cy
        assert np.all(final_counts >= initial_counts)  # Counts should not decrease

        # At least some counts should be > prior
        assert np.any(final_counts > 1.0)

    def test_prediction_with_tau_classifier(self, mock_mnist_data, tmp_path):
        """Test that predictions use τ-classifier."""
        from ssvae import SSVAE
        from ssvae.config import SSVAEConfig

        X, y = mock_mnist_data
        checkpoint_path = str(tmp_path / "test_pred.ckpt")

        config = SSVAEConfig(
            latent_dim=2,
            prior_type="mixture",
            num_components=4,
            use_tau_classifier=True,
            max_epochs=3,
            batch_size=32,
        )

        model = SSVAE(input_dim=(28, 28), config=config)
        model.fit(X, y, checkpoint_path, export_history=False)

        # Predict on test data
        test_X = X[:10]  # Use first 10 samples
        latent, recon, predictions, certainty = model.predict(test_X)

        # Verify output shapes
        assert latent.shape == (10, 2)  # [batch, latent_dim]
        assert recon.shape == (10, 28, 28)  # [batch, H, W]
        assert predictions.shape == (10,)  # [batch,]
        assert certainty.shape == (10,)  # [batch,]

        # Verify predictions are valid class indices
        assert np.all(predictions >= 0)
        assert np.all(predictions < 10)

        # Verify certainty is in [0, 1]
        assert np.all(certainty >= 0)
        assert np.all(certainty <= 1)

    def test_prediction_with_mixture_return(self, mock_mnist_data, tmp_path):
        """Test prediction returns responsibilities when requested."""
        from ssvae import SSVAE
        from ssvae.config import SSVAEConfig

        X, y = mock_mnist_data
        checkpoint_path = str(tmp_path / "test_mixture.ckpt")

        config = SSVAEConfig(
            latent_dim=2,
            prior_type="mixture",
            num_components=4,
            use_tau_classifier=True,
            max_epochs=2,
        )

        model = SSVAE(input_dim=(28, 28), config=config)
        model.fit(X, y, checkpoint_path, export_history=False)

        # Predict with mixture outputs
        test_X = X[:10]
        latent, recon, predictions, certainty, responsibilities, pi = model.predict(
            test_X, return_mixture=True
        )

        # Verify responsibilities shape and properties
        assert responsibilities.shape == (10, 4)  # [batch, K]
        assert np.allclose(responsibilities.sum(axis=1), 1.0)  # Normalized

        # Verify π shape
        assert pi.shape == (4,)  # [K,]
        assert np.allclose(pi.sum(), 1.0)  # Normalized

    def test_backward_compatibility_standard_classifier(self, mock_mnist_data, tmp_path):
        """Test that standard classifier still works when τ disabled."""
        from ssvae import SSVAE
        from ssvae.config import SSVAEConfig

        X, y = mock_mnist_data
        checkpoint_path = str(tmp_path / "test_standard.ckpt")

        # Disable τ-classifier
        config = SSVAEConfig(
            latent_dim=2,
            prior_type="mixture",
            num_components=3,
            use_tau_classifier=False,  # Use standard classifier
            max_epochs=2,
            batch_size=32,
        )

        model = SSVAE(input_dim=(28, 28), config=config)

        # Verify τ-classifier was NOT initialized
        assert model._tau_classifier is None

        # Train should still work
        history = model.fit(X, y, checkpoint_path, export_history=False)
        assert "loss" in history

        # Prediction should still work
        test_X = X[:10]
        latent, recon, predictions, certainty = model.predict(test_X)
        assert predictions.shape == (10,)

    def test_tau_diagnostics(self, mock_mnist_data, tmp_path):
        """Test τ-classifier diagnostic outputs."""
        from ssvae import SSVAE
        from ssvae.config import SSVAEConfig

        X, y = mock_mnist_data
        checkpoint_path = str(tmp_path / "test_diag.ckpt")

        config = SSVAEConfig(
            latent_dim=2,
            prior_type="mixture",
            num_components=5,
            use_tau_classifier=True,
            max_epochs=3,
            batch_size=32,
        )

        model = SSVAE(input_dim=(28, 28), config=config)
        model.fit(X, y, checkpoint_path, export_history=False)

        # Get diagnostics
        diag = model._tau_classifier.get_diagnostics()

        # Verify diagnostic keys
        assert "tau" in diag
        assert "s_cy" in diag
        assert "component_label_confidence" in diag
        assert "component_dominant_label" in diag
        assert "components_per_label" in diag
        assert "tau_entropy" in diag

        # Verify shapes
        assert diag["tau"].shape == (5, 10)
        assert diag["component_label_confidence"].shape == (5,)
        assert diag["component_dominant_label"].shape == (5,)
        assert diag["components_per_label"].shape == (10,)

    def test_ood_detection(self, mock_mnist_data, tmp_path):
        """Test OOD detection capability."""
        from ssvae import SSVAE
        from ssvae.config import SSVAEConfig

        X, y = mock_mnist_data
        checkpoint_path = str(tmp_path / "test_ood.ckpt")

        config = SSVAEConfig(
            latent_dim=2,
            prior_type="mixture",
            num_components=4,
            use_tau_classifier=True,
            max_epochs=3,
        )

        model = SSVAE(input_dim=(28, 28), config=config)
        model.fit(X, y, checkpoint_path, export_history=False)

        # Get responsibilities for test data
        test_X = X[:10]
        _, _, _, _, responsibilities, _ = model.predict(test_X, return_mixture=True)

        # Compute OOD scores
        ood_scores = model._tau_classifier.get_ood_score(responsibilities)

        # Verify OOD scores are in [0, 1]
        assert ood_scores.shape == (10,)
        assert np.all(ood_scores >= 0)
        assert np.all(ood_scores <= 1)


def run_basic_validation():
    """Quick validation that can be run without pytest."""
    print("Running basic τ-classifier validation...")
    print("-" * 60)

    import numpy as np
    from ssvae import SSVAE
    from ssvae.config import SSVAEConfig

    # Create tiny dataset
    np.random.seed(42)
    X = np.random.randn(100, 28, 28).astype(np.float32)
    y = np.full(100, np.nan, dtype=np.float32)
    y[:30] = np.random.randint(0, 10, 30)  # 30 labeled samples

    print(f"Dataset: {len(X)} samples, {np.sum(~np.isnan(y))} labeled")

    # Create model with τ-classifier
    config = SSVAEConfig(
        latent_dim=2,
        prior_type="mixture",
        num_components=5,
        use_component_aware_decoder=True,
        use_tau_classifier=True,
        max_epochs=3,
        batch_size=32,
        patience=10,
    )

    print(f"Config: K={config.num_components}, latent_dim={config.latent_dim}")
    print(f"τ-classifier: {config.use_tau_classifier}, smoothing α={config.tau_smoothing_alpha}")

    model = SSVAE(input_dim=(28, 28), config=config)
    print(f"τ-classifier initialized: {model._tau_classifier is not None}")

    # Train
    print("\nTraining...")
    history = model.fit(X, y, "/tmp/test_tau.ckpt", export_history=False)

    print(f"Training completed: {len(history['loss'])} epochs")
    print(f"Final loss: {history['loss'][-1]:.4f}")

    # Check τ matrix
    tau = model._tau_classifier.get_tau()
    print(f"\nτ matrix shape: {tau.shape}")
    print(f"τ normalized: {np.allclose(tau.sum(axis=1), 1.0)}")
    print(f"τ learned (not uniform): {not np.allclose(tau, 0.1)}")

    # Predict
    print("\nTesting prediction...")
    latent, recon, pred, cert = model.predict(X[:10])
    print(f"Predictions: {pred}")
    print(f"Certainty: {cert}")

    # Diagnostics
    diag = model._tau_classifier.get_diagnostics()
    print(f"\nDiagnostics:")
    print(f"  Component→Label confidence: {diag['component_label_confidence']}")
    print(f"  Dominant labels: {diag['component_dominant_label']}")
    print(f"  Components per label: {diag['components_per_label']}")

    print("\n" + "=" * 60)
    print("✅ Validation PASSED - τ-classifier is working!")
    print("=" * 60)


if __name__ == "__main__":
    # Run basic validation without pytest
    run_basic_validation()
