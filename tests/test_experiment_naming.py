"""Unit tests for experiment naming system.

Tests the architecture code generation logic that creates human-readable
experiment directory names from SSVAE configurations.
"""
import pytest

from rcmvae.domain.config import SSVAEConfig
from use_cases.experiments.src.naming import (
    generate_architecture_code,
    generate_naming_legend,
    _encode_prior,
    _encode_classifier,
    _encode_decoder,
)


class TestPriorEncoding:
    """Test prior type encoding."""

    def test_standard_prior(self):
        """Standard Gaussian prior should encode as 'std'."""
        config = SSVAEConfig(prior_type="standard")
        assert _encode_prior(config) == "std"

    def test_mixture_prior_basic(self):
        """Mixture prior encodes with component count."""
        config = SSVAEConfig(prior_type="mixture", num_components=10)
        assert _encode_prior(config) == "mix10"

    def test_mixture_prior_with_dirichlet(self):
        """Mixture with Dirichlet adds -dir modifier."""
        config = SSVAEConfig(
            prior_type="mixture",
            num_components=10,
            dirichlet_alpha=5.0
        )
        assert _encode_prior(config) == "mix10-dir"

    def test_mixture_prior_without_dirichlet(self):
        """Mixture without Dirichlet (None or 0) has no modifier."""
        config1 = SSVAEConfig(
            prior_type="mixture",
            num_components=10,
            dirichlet_alpha=None
        )
        assert _encode_prior(config1) == "mix10"

        config2 = SSVAEConfig(
            prior_type="mixture",
            num_components=10,
            dirichlet_alpha=0.0
        )
        assert _encode_prior(config2) == "mix10"

    def test_vamp_prior_kmeans(self):
        """VampPrior with k-means initialization."""
        config = SSVAEConfig(
            prior_type="vamp",
            num_components=20,
            vamp_pseudo_init_method="kmeans"
        )
        assert _encode_prior(config) == "vamp20-km"

    def test_vamp_prior_random(self):
        """VampPrior with random initialization."""
        config = SSVAEConfig(
            prior_type="vamp",
            num_components=20,
            vamp_pseudo_init_method="random"
        )
        assert _encode_prior(config) == "vamp20-rand"

    def test_vamp_prior_invalid_init(self):
        """VampPrior with invalid init method raises error."""
        config = SSVAEConfig(
            prior_type="vamp",
            num_components=20,
            vamp_pseudo_init_method="invalid"
        )
        with pytest.raises(ValueError, match="VampPrior requires"):
            _encode_prior(config)

    def test_geometric_mog_circle(self):
        """Geometric MoG with circle arrangement."""
        config = SSVAEConfig(
            prior_type="geometric_mog",
            num_components=8,
            geometric_arrangement="circle"
        )
        assert _encode_prior(config) == "geo8-circle"

    def test_geometric_mog_grid(self):
        """Geometric MoG with grid arrangement."""
        config = SSVAEConfig(
            prior_type="geometric_mog",
            num_components=9,
            geometric_arrangement="grid"
        )
        assert _encode_prior(config) == "geo9-grid"

    def test_geometric_mog_invalid_arrangement(self):
        """Geometric MoG with invalid arrangement raises error."""
        with pytest.raises(ValueError, match="Geometric MoG requires"):
            SSVAEConfig(
                prior_type="geometric_mog",
                num_components=9,
                geometric_arrangement="invalid"
            )

    def test_unknown_prior_type(self):
        """Unknown prior type raises error."""
        config = SSVAEConfig()
        config.prior_type = "unknown"
        with pytest.raises(ValueError, match="Unknown prior_type"):
            _encode_prior(config)


class TestClassifierEncoding:
    """Test classifier type encoding."""

    def test_tau_classifier(self):
        """τ-classifier encodes as 'tau'."""
        config = SSVAEConfig(
            prior_type="mixture",
            use_tau_classifier=True
        )
        assert _encode_classifier(config) == "tau"

    def test_standard_head(self):
        """Standard classifier head encodes as 'head'."""
        config = SSVAEConfig(use_tau_classifier=False)
        assert _encode_classifier(config) == "head"


class TestDecoderEncoding:
    """Test decoder feature encoding."""

    def test_plain_decoder(self):
        """Plain decoder with no features."""
        config = SSVAEConfig(
            use_component_aware_decoder=False,
            use_heteroscedastic_decoder=False
        )
        assert _encode_decoder(config) == "plain"

    def test_component_aware_only(self):
        """Component-aware decoder only."""
        config = SSVAEConfig(
            prior_type="mixture",
            use_component_aware_decoder=True,
            use_heteroscedastic_decoder=False
        )
        assert _encode_decoder(config) == "ca"

    def test_heteroscedastic_only(self):
        """Heteroscedastic decoder only."""
        config = SSVAEConfig(
            use_component_aware_decoder=False,
            use_heteroscedastic_decoder=True
        )
        assert _encode_decoder(config) == "het"

    def test_component_aware_and_heteroscedastic(self):
        """Both component-aware and heteroscedastic."""
        config = SSVAEConfig(
            prior_type="mixture",
            use_component_aware_decoder=True,
            use_heteroscedastic_decoder=True
        )
        assert _encode_decoder(config) == "ca-het"


class TestFullArchitectureCode:
    """Test complete architecture code generation."""

    def test_standard_vae(self):
        """Standard VAE with no special features."""
        config = SSVAEConfig(
            prior_type="standard",
            use_tau_classifier=False,
            use_component_aware_decoder=False,
            use_heteroscedastic_decoder=False
        )
        assert generate_architecture_code(config) == "std_head_plain"

    def test_full_mixture_model(self):
        """Full mixture model with all features."""
        config = SSVAEConfig(
            prior_type="mixture",
            num_components=10,
            dirichlet_alpha=5.0,
            use_tau_classifier=True,
            use_component_aware_decoder=True,
            use_heteroscedastic_decoder=True
        )
        assert generate_architecture_code(config) == "mix10-dir_tau_ca-het"

    def test_vamp_prior_configuration(self):
        """VampPrior with typical settings."""
        config = SSVAEConfig(
            prior_type="vamp",
            num_components=20,
            vamp_pseudo_init_method="kmeans",
            use_tau_classifier=True,
            use_component_aware_decoder=True,
            use_heteroscedastic_decoder=False
        )
        assert generate_architecture_code(config) == "vamp20-km_tau_ca"

    def test_geometric_debug_configuration(self):
        """Geometric MoG for debugging."""
        config = SSVAEConfig(
            prior_type="geometric_mog",
            num_components=10,
            geometric_arrangement="circle",
            use_tau_classifier=True,
            use_component_aware_decoder=False,
            use_heteroscedastic_decoder=False
        )
        assert generate_architecture_code(config) == "geo10-circle_tau_plain"

    def test_mixture_without_tau(self):
        """Mixture prior with standard classifier head."""
        config = SSVAEConfig(
            prior_type="mixture",
            num_components=10,
            use_tau_classifier=False,
            use_component_aware_decoder=True,
            use_heteroscedastic_decoder=True
        )
        assert generate_architecture_code(config) == "mix10_head_ca-het"


class TestNamingLegend:
    """Test naming legend generation."""

    def test_legend_generation(self):
        """Legend should be non-empty markdown."""
        legend = generate_naming_legend()

        assert isinstance(legend, str)
        assert len(legend) > 100  # Should be substantial
        assert "# Experiment Naming Legend" in legend
        assert "Architecture Code Structure" in legend
        assert "Validation Rules" in legend

    def test_legend_has_timestamp(self):
        """Legend should include generation timestamp."""
        legend = generate_naming_legend()
        assert "Auto-generated:" in legend

    def test_legend_has_examples(self):
        """Legend should include usage examples."""
        legend = generate_naming_legend()
        assert "baseline__mix10-dir_tau_ca-het" in legend
        assert "quick__std_head_plain" in legend

    def test_legend_has_all_prior_types(self):
        """Legend should document all prior types."""
        legend = generate_naming_legend()
        assert "std" in legend
        assert "mix" in legend
        assert "vamp" in legend
        assert "geo" in legend

    def test_legend_has_validation_rules(self):
        """Legend should document validation constraints."""
        legend = generate_naming_legend()
        assert "τ-classifier requires mixture prior" in legend or \
               "tau" in legend.lower() and "mixture" in legend.lower()


class TestComponentCountVariations:
    """Test naming with different component counts."""

    @pytest.mark.parametrize("num_components", [5, 10, 20, 50, 100])
    def test_mixture_component_counts(self, num_components):
        """Test mixture naming with various K values."""
        config = SSVAEConfig(
            prior_type="mixture",
            num_components=num_components
        )
        code = _encode_prior(config)
        assert code == f"mix{num_components}"

    @pytest.mark.parametrize("num_components", [4, 9, 16, 25])
    def test_geometric_grid_perfect_squares(self, num_components):
        """Test geometric grid with perfect square K."""
        config = SSVAEConfig(
            prior_type="geometric_mog",
            num_components=num_components,
            geometric_arrangement="grid"
        )
        code = _encode_prior(config)
        assert code == f"geo{num_components}-grid"
