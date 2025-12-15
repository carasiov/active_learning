"""Unit tests for experiment configuration validation.

Tests the validation rules that enforce architectural constraints and catch
invalid configuration combinations before training starts.
"""
import warnings

import pytest

from rcmvae.domain.config import SSVAEConfig
from use_cases.experiments.src.validation import (
    ConfigValidationError,
    validate_config,
    validate_hyperparameters,
    _validate_tau_classifier,
    _validate_component_aware_decoder,
    _validate_vamp_prior,
    _validate_geometric_mog,
    _validate_heteroscedastic_decoder,
)


class TestTauClassifierValidation:
    """Test τ-classifier validation rules."""

    def test_tau_with_mixture_prior_valid(self):
        """τ-classifier with mixture prior should pass."""
        config = SSVAEConfig(
            prior_type="mixture",
            num_components=10,
            use_tau_classifier=True
        )
        _validate_tau_classifier(config)  # Should not raise

    def test_tau_with_vamp_prior_valid(self):
        """τ-classifier with VampPrior should pass."""
        config = SSVAEConfig(
            prior_type="vamp",
            num_components=10,
            vamp_pseudo_init_method="kmeans",
            use_tau_classifier=True
        )
        _validate_tau_classifier(config)  # Should not raise

    def test_tau_with_geometric_mog_valid(self):
        """τ-classifier with geometric MoG should pass."""
        config = SSVAEConfig(
            prior_type="geometric_mog",
            num_components=16,
            geometric_arrangement="grid",
            use_tau_classifier=True
        )
        _validate_tau_classifier(config)  # Should not raise

    def test_tau_with_standard_prior_invalid(self):
        """τ-classifier with standard prior should fail."""
        with pytest.raises(ValueError, match="τ-classifier.*requires mixture-based"):
            SSVAEConfig(
                prior_type="standard",
                use_tau_classifier=True,
            )

    def test_tau_disabled_with_any_prior_valid(self):
        """Disabled τ-classifier should pass with any prior."""
        for prior_type in ["standard", "mixture", "vamp", "geometric_mog"]:
            config = SSVAEConfig(
                prior_type=prior_type,
                use_tau_classifier=False
            )
            if prior_type == "vamp":
                config.vamp_pseudo_init_method = "kmeans"
            if prior_type == "geometric_mog":
                config.geometric_arrangement = "circle"

            _validate_tau_classifier(config)  # Should not raise


class TestComponentAwareDecoderValidation:
    """Test component-aware decoder validation rules."""

    def test_component_aware_with_mixture_prior_valid(self):
        """Component-aware with mixture prior should pass."""
        config = SSVAEConfig(
            prior_type="mixture",
            num_components=10,
            use_component_aware_decoder=True
        )
        _validate_component_aware_decoder(config)  # Should not raise

    def test_component_aware_with_vamp_prior_valid(self):
        """Component-aware with VampPrior should pass."""
        config = SSVAEConfig(
            prior_type="vamp",
            num_components=10,
            vamp_pseudo_init_method="kmeans",
            use_component_aware_decoder=True
        )
        _validate_component_aware_decoder(config)  # Should not raise

    def test_component_aware_with_geometric_mog_valid(self):
        """Component-aware with geometric MoG should pass."""
        config = SSVAEConfig(
            prior_type="geometric_mog",
            num_components=9,
            geometric_arrangement="grid",
            use_component_aware_decoder=True
        )
        _validate_component_aware_decoder(config)  # Should not raise

    def test_component_aware_with_standard_prior_invalid(self):
        """Component-aware with standard prior should fail."""
        config = SSVAEConfig(
            prior_type="standard",
            use_component_aware_decoder=True
        )
        with pytest.raises(ConfigValidationError, match="Component-aware decoder.*requires mixture-based prior"):
            _validate_component_aware_decoder(config)

    def test_component_aware_disabled_with_any_prior_valid(self):
        """Disabled component-aware should pass with any prior."""
        for prior_type in ["standard", "mixture", "vamp", "geometric_mog"]:
            config = SSVAEConfig(
                prior_type=prior_type,
                use_component_aware_decoder=False
            )
            if prior_type == "vamp":
                config.vamp_pseudo_init_method = "kmeans"
            if prior_type == "geometric_mog":
                config.geometric_arrangement = "circle"

            _validate_component_aware_decoder(config)  # Should not raise


class TestVampPriorValidation:
    """Test VampPrior validation rules."""

    def test_vamp_with_kmeans_init_valid(self):
        """VampPrior with k-means initialization should pass."""
        config = SSVAEConfig(
            prior_type="vamp",
            num_components=20,
            vamp_pseudo_init_method="kmeans"
        )
        _validate_vamp_prior(config)  # Should not raise

    def test_vamp_with_random_init_valid(self):
        """VampPrior with random initialization should pass."""
        config = SSVAEConfig(
            prior_type="vamp",
            num_components=20,
            vamp_pseudo_init_method="random"
        )
        _validate_vamp_prior(config)  # Should not raise

    def test_vamp_with_invalid_init_method(self):
        """VampPrior with invalid init method should fail."""
        config = SSVAEConfig(
            prior_type="vamp",
            num_components=20,
            vamp_pseudo_init_method="invalid"
        )
        with pytest.raises(ConfigValidationError, match="VampPrior requires"):
            _validate_vamp_prior(config)

    def test_vamp_with_negative_num_samples(self):
        """VampPrior with negative num_samples_kl should fail."""
        config = SSVAEConfig(
            prior_type="vamp",
            num_components=20,
            vamp_pseudo_init_method="kmeans",
            vamp_num_samples_kl=1,
        )
        config.vamp_num_samples_kl = 0
        with pytest.raises(ConfigValidationError, match="vamp_num_samples_kl must be >= 1"):
            _validate_vamp_prior(config)

    def test_vamp_with_high_num_samples_warns(self):
        """VampPrior with high num_samples_kl should warn."""
        config = SSVAEConfig(
            prior_type="vamp",
            num_components=20,
            vamp_pseudo_init_method="kmeans",
            vamp_num_samples_kl=20
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_vamp_prior(config)
            assert len(w) >= 1
            assert "vamp_num_samples_kl" in str(w[0].message)

    def test_vamp_with_high_lr_scale_warns(self):
        """VampPrior with LR scale >= 1.0 should warn."""
        config = SSVAEConfig(
            prior_type="vamp",
            num_components=20,
            vamp_pseudo_init_method="kmeans",
            vamp_pseudo_lr_scale=0.1,
        )
        config.vamp_pseudo_lr_scale = 1.5
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_vamp_prior(config)
            assert len(w) >= 1
            assert "vamp_pseudo_lr_scale" in str(w[0].message)

    def test_non_vamp_prior_skips_validation(self):
        """Non-VampPrior configs should skip VampPrior validation."""
        config = SSVAEConfig(prior_type="mixture")
        _validate_vamp_prior(config)  # Should not raise or warn


class TestGeometricMoGValidation:
    """Test Geometric MoG validation rules."""

    def test_geometric_with_circle_valid(self):
        """Geometric MoG with circle arrangement should pass."""
        config = SSVAEConfig(
            prior_type="geometric_mog",
            num_components=8,
            geometric_arrangement="circle"
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _validate_geometric_mog(config)  # Should not raise error

    def test_geometric_with_grid_valid(self):
        """Geometric MoG with grid arrangement (perfect square) should pass."""
        config = SSVAEConfig(
            prior_type="geometric_mog",
            num_components=9,
            geometric_arrangement="grid"
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _validate_geometric_mog(config)  # Should not raise error

    def test_geometric_with_invalid_arrangement(self):
        """Geometric MoG with invalid arrangement should fail."""
        config = SSVAEConfig(
            prior_type="geometric_mog",
            num_components=9,
            geometric_arrangement="circle"
        )
        # Manually override after construction to bypass dataclass validation.
        config.geometric_arrangement = "invalid"
        with pytest.raises(ConfigValidationError, match="Geometric MoG requires"):
            _validate_geometric_mog(config)

    def test_geometric_grid_with_non_perfect_square(self):
        """Geometric grid with non-perfect-square K should fail."""
        config = SSVAEConfig(
            prior_type="geometric_mog",
            num_components=10,
            geometric_arrangement="grid"
        )
        with pytest.raises(ConfigValidationError, match="perfect square"):
            _validate_geometric_mog(config)

    @pytest.mark.parametrize("num_components", [4, 9, 16, 25, 36])
    def test_geometric_grid_with_perfect_squares(self, num_components):
        """Geometric grid with perfect squares should pass."""
        config = SSVAEConfig(
            prior_type="geometric_mog",
            num_components=num_components,
            geometric_arrangement="grid"
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _validate_geometric_mog(config)  # Should not raise error

    def test_geometric_with_small_radius_warns(self):
        """Geometric MoG with very small radius should warn."""
        config = SSVAEConfig(
            prior_type="geometric_mog",
            num_components=9,
            geometric_arrangement="grid",
            geometric_radius=0.3
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_geometric_mog(config)
            assert any("geometric_radius" in str(warning.message) for warning in w)

    def test_geometric_with_large_radius_warns(self):
        """Geometric MoG with very large radius should warn."""
        config = SSVAEConfig(
            prior_type="geometric_mog",
            num_components=9,
            geometric_arrangement="grid",
            geometric_radius=10.0
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_geometric_mog(config)
            assert any("geometric_radius" in str(warning.message) for warning in w)

    def test_geometric_always_warns_about_topology(self):
        """Geometric MoG should always warn about induced topology."""
        config = SSVAEConfig(
            prior_type="geometric_mog",
            num_components=9,
            geometric_arrangement="grid"
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_geometric_mog(config)
            assert any("topology" in str(warning.message).lower() for warning in w)

    def test_non_geometric_prior_skips_validation(self):
        """Non-geometric configs should skip geometric validation."""
        config = SSVAEConfig(prior_type="mixture")
        _validate_geometric_mog(config)  # Should not raise or warn


class TestHeteroscedasticDecoderValidation:
    """Test heteroscedastic decoder validation rules."""

    def test_heteroscedastic_with_valid_range(self):
        """Heteroscedastic with valid sigma range should pass."""
        config = SSVAEConfig(
            use_heteroscedastic_decoder=True,
            sigma_min=0.05,
            sigma_max=0.5
        )
        _validate_heteroscedastic_decoder(config)  # Should not raise

    def test_heteroscedastic_with_narrow_range_warns(self):
        """Heteroscedastic with narrow range should warn."""
        config = SSVAEConfig(
            use_heteroscedastic_decoder=True,
            sigma_min=0.1,
            sigma_max=0.15
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_heteroscedastic_decoder(config)
            assert len(w) >= 1
            assert "narrow" in str(w[0].message).lower()

    def test_heteroscedastic_with_wide_range_warns(self):
        """Heteroscedastic with very wide range should warn."""
        config = SSVAEConfig(
            use_heteroscedastic_decoder=True,
            sigma_min=0.001,
            sigma_max=10.0
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_heteroscedastic_decoder(config)
            assert len(w) >= 1
            assert "wide" in str(w[0].message).lower()

    def test_heteroscedastic_disabled_skips_validation(self):
        """Disabled heteroscedastic should skip validation."""
        config = SSVAEConfig(use_heteroscedastic_decoder=False)
        _validate_heteroscedastic_decoder(config)  # Should not raise or warn


class TestFullConfigValidation:
    """Test the main validate_config function with complete configurations."""

    def test_valid_standard_vae(self):
        """Standard VAE configuration should pass all validation."""
        config = SSVAEConfig(
            prior_type="standard",
            use_tau_classifier=False,
            use_component_aware_decoder=False
        )
        validate_config(config)  # Should not raise

    def test_valid_full_mixture_model(self):
        """Full mixture model should pass all validation."""
        config = SSVAEConfig(
            prior_type="mixture",
            num_components=10,
            dirichlet_alpha=5.0,
            use_tau_classifier=True,
            use_component_aware_decoder=True,
            use_heteroscedastic_decoder=True
        )
        validate_config(config)  # Should not raise

    def test_valid_vamp_configuration(self):
        """VampPrior configuration should pass all validation."""
        config = SSVAEConfig(
            prior_type="vamp",
            num_components=20,
            vamp_pseudo_init_method="kmeans",
            use_tau_classifier=True,
            use_component_aware_decoder=True
        )
        validate_config(config)  # Should not raise

    def test_invalid_tau_with_standard_prior(self):
        """Config with τ-classifier + standard prior should fail validation."""
        with pytest.raises(ValueError, match="τ-classifier.*requires mixture-based"):
            SSVAEConfig(
                prior_type="standard",
                use_tau_classifier=True,
            )

    def test_invalid_component_aware_with_standard_prior(self):
        """Config with component-aware + standard prior should fail."""
        config = SSVAEConfig(
            prior_type="standard",
            use_component_aware_decoder=True
        )
        with pytest.raises(ConfigValidationError):
            validate_config(config)


class TestHyperparameterValidation:
    """Test hyperparameter validation (warnings only)."""

    def test_high_kl_weight_warns(self):
        """High KL weight should generate warning."""
        config = SSVAEConfig(kl_weight=20.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_hyperparameters(config)
            assert any("kl_weight" in str(warning.message) for warning in w)

    def test_low_kl_weight_warns(self):
        """Very low KL weight should generate warning."""
        config = SSVAEConfig(kl_weight=0.001)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_hyperparameters(config)
            assert any("kl_weight" in str(warning.message) for warning in w)

    def test_positive_diversity_weight_warns(self):
        """Positive diversity weight (discourages diversity) should warn."""
        config = SSVAEConfig(
            prior_type="mixture",
            component_diversity_weight=0.05
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_hyperparameters(config)
            assert any("component_diversity_weight" in str(warning.message) for warning in w)

    def test_high_learning_rate_warns(self):
        """High learning rate should generate warning."""
        config = SSVAEConfig(learning_rate=0.1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_hyperparameters(config)
            assert any("learning_rate" in str(warning.message) for warning in w)

    def test_small_batch_size_warns(self):
        """Small batch size should generate warning."""
        config = SSVAEConfig(batch_size=16)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_hyperparameters(config)
            assert any("batch_size" in str(warning.message) for warning in w)
