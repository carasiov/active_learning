"""Comprehensive tests for the modular config system.

Tests cover:
- Base configs (NetworkConfig, TrainingConfig, LossConfig, DecoderFeatures)
- Prior configs (Standard, Mixture, Vamp, GeometricMoG)
- ExperimentConfig composition
- Converters (dict ↔ ExperimentConfig)
"""

import pytest

from rcmvae.config import (
    DecoderFeatures,
    ExperimentConfig,
    GeometricMoGPriorConfig,
    LossConfig,
    MixturePriorConfig,
    NetworkConfig,
    PriorConfig,
    StandardPriorConfig,
    TrainingConfig,
    VampPriorConfig,
    experiment_config_from_dict,
)


# ============================================================================
# NetworkConfig Tests
# ============================================================================


def test_network_config_defaults():
    """Test NetworkConfig default values."""
    config = NetworkConfig()
    assert config.num_classes == 10
    assert config.latent_dim == 2
    assert config.hidden_dims == (256, 128, 64)
    assert config.encoder_type == "dense"
    assert config.decoder_type == "dense"
    assert config.classifier_type == "dense"
    assert config.input_hw is None
    assert config.dropout_rate == 0.2


def test_network_config_custom():
    """Test NetworkConfig with custom values."""
    config = NetworkConfig(
        num_classes=5,
        latent_dim=10,
        hidden_dims=(512, 256),
        encoder_type="conv",
        decoder_type="conv",
        dropout_rate=0.3,
    )
    assert config.num_classes == 5
    assert config.latent_dim == 10
    assert config.hidden_dims == (512, 256)
    assert config.encoder_type == "conv"
    assert config.decoder_type == "conv"
    assert config.dropout_rate == 0.3


def test_network_config_validation():
    """Test NetworkConfig validation."""
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        NetworkConfig(latent_dim=0)

    with pytest.raises(ValueError, match="num_classes must be positive"):
        NetworkConfig(num_classes=-1)

    with pytest.raises(ValueError, match="dropout_rate must be in"):
        NetworkConfig(dropout_rate=1.5)

    with pytest.raises(ValueError, match="encoder_type must be one of"):
        NetworkConfig(encoder_type="invalid")


# ============================================================================
# TrainingConfig Tests
# ============================================================================


def test_training_config_defaults():
    """Test TrainingConfig default values."""
    config = TrainingConfig()
    assert config.learning_rate == 1e-3
    assert config.batch_size == 128
    assert config.max_epochs == 300
    assert config.patience == 50
    assert config.val_split == 0.1
    assert config.random_seed == 42
    assert config.grad_clip_norm == 1.0
    assert config.weight_decay == 1e-4
    assert config.monitor_metric == "classification_loss"


def test_training_config_custom():
    """Test TrainingConfig with custom values."""
    config = TrainingConfig(
        learning_rate=1e-4,
        batch_size=256,
        max_epochs=100,
        patience=20,
        val_split=0.2,
    )
    assert config.learning_rate == 1e-4
    assert config.batch_size == 256
    assert config.max_epochs == 100
    assert config.patience == 20
    assert config.val_split == 0.2


def test_training_config_validation():
    """Test TrainingConfig validation."""
    with pytest.raises(ValueError, match="learning_rate must be positive"):
        TrainingConfig(learning_rate=0)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        TrainingConfig(batch_size=-1)

    with pytest.raises(ValueError, match="val_split must be in"):
        TrainingConfig(val_split=1.5)


# ============================================================================
# LossConfig Tests
# ============================================================================


def test_loss_config_defaults():
    """Test LossConfig default values."""
    config = LossConfig()
    assert config.reconstruction_loss == "mse"
    assert config.recon_weight == 500.0
    assert config.kl_weight == 5.0
    assert config.label_weight == 0.0
    assert config.use_contrastive is False
    assert config.contrastive_weight == 0.0


def test_loss_config_custom():
    """Test LossConfig with custom values."""
    config = LossConfig(
        reconstruction_loss="bce",
        recon_weight=1.0,
        kl_weight=10.0,
        use_contrastive=True,
        contrastive_weight=0.5,
    )
    assert config.reconstruction_loss == "bce"
    assert config.recon_weight == 1.0
    assert config.kl_weight == 10.0
    assert config.use_contrastive is True
    assert config.contrastive_weight == 0.5


def test_loss_config_validation():
    """Test LossConfig validation."""
    with pytest.raises(ValueError, match="reconstruction_loss must be one of"):
        LossConfig(reconstruction_loss="invalid")

    with pytest.raises(ValueError, match="recon_weight must be non-negative"):
        LossConfig(recon_weight=-1.0)


# ============================================================================
# DecoderFeatures Tests
# ============================================================================


def test_decoder_features_defaults():
    """Test DecoderFeatures default values."""
    config = DecoderFeatures()
    assert config.use_heteroscedastic_decoder is False
    assert config.sigma_min == 0.05
    assert config.sigma_max == 0.5
    assert config.use_component_aware_decoder is True
    assert config.component_embedding_dim is None
    assert config.top_m_gating == 0
    assert config.soft_embedding_warmup_epochs == 0


def test_decoder_features_custom():
    """Test DecoderFeatures with custom values."""
    config = DecoderFeatures(
        use_heteroscedastic_decoder=True,
        sigma_min=0.1,
        sigma_max=1.0,
        component_embedding_dim=16,
        top_m_gating=5,
    )
    assert config.use_heteroscedastic_decoder is True
    assert config.sigma_min == 0.1
    assert config.sigma_max == 1.0
    assert config.component_embedding_dim == 16
    assert config.top_m_gating == 5


def test_decoder_features_validation():
    """Test DecoderFeatures validation."""
    with pytest.raises(ValueError, match="sigma_min must be positive"):
        DecoderFeatures(sigma_min=0)

    with pytest.raises(ValueError, match="sigma_max.*must be greater than"):
        DecoderFeatures(sigma_min=0.5, sigma_max=0.3)

    with pytest.raises(ValueError, match="top_m_gating must be non-negative"):
        DecoderFeatures(top_m_gating=-1)


# ============================================================================
# Prior Config Tests
# ============================================================================


def test_standard_prior_config():
    """Test StandardPriorConfig."""
    config = StandardPriorConfig()
    assert config.get_prior_type() == "standard"


def test_mixture_prior_config_defaults():
    """Test MixturePriorConfig default values."""
    config = MixturePriorConfig()
    assert config.num_components == 10
    assert config.kl_c_weight == 1.0
    assert config.kl_c_anneal_epochs == 0
    assert config.learnable_pi is False
    assert config.dirichlet_alpha is None
    assert config.component_diversity_weight == 0.0
    assert config.use_tau_classifier is False
    assert config.tau_smoothing_alpha == 1.0
    assert config.get_prior_type() == "mixture"


def test_mixture_prior_config_custom():
    """Test MixturePriorConfig with custom values."""
    config = MixturePriorConfig(
        num_components=20,
        kl_c_weight=2.0,
        kl_c_anneal_epochs=10,
        learnable_pi=True,
        use_tau_classifier=True,
    )
    assert config.num_components == 20
    assert config.kl_c_weight == 2.0
    assert config.kl_c_anneal_epochs == 10
    assert config.learnable_pi is True
    assert config.use_tau_classifier is True


def test_mixture_prior_config_validation():
    """Test MixturePriorConfig validation."""
    with pytest.raises(ValueError, match="num_components must be positive"):
        MixturePriorConfig(num_components=0)

    with pytest.raises(ValueError, match="kl_c_weight must be non-negative"):
        MixturePriorConfig(kl_c_weight=-1.0)

    with pytest.raises(ValueError, match="tau_smoothing_alpha must be positive"):
        MixturePriorConfig(tau_smoothing_alpha=0)


def test_vamp_prior_config_defaults():
    """Test VampPriorConfig default values."""
    config = VampPriorConfig()
    assert config.num_components == 10
    assert config.vamp_num_samples_kl == 1
    assert config.vamp_pseudo_lr_scale == 0.1
    assert config.vamp_pseudo_init_method == "random"
    assert config.get_prior_type() == "vamp"


def test_vamp_prior_config_custom():
    """Test VampPriorConfig with custom values."""
    config = VampPriorConfig(
        num_components=15,
        vamp_num_samples_kl=5,
        vamp_pseudo_lr_scale=0.05,
        vamp_pseudo_init_method="kmeans",
    )
    assert config.num_components == 15
    assert config.vamp_num_samples_kl == 5
    assert config.vamp_pseudo_lr_scale == 0.05
    assert config.vamp_pseudo_init_method == "kmeans"


def test_vamp_prior_config_validation():
    """Test VampPriorConfig validation."""
    with pytest.raises(ValueError, match="vamp_num_samples_kl must be >= 1"):
        VampPriorConfig(vamp_num_samples_kl=0)

    with pytest.raises(ValueError, match="vamp_pseudo_lr_scale must be in"):
        VampPriorConfig(vamp_pseudo_lr_scale=1.5)


def test_geometric_mog_prior_config_defaults():
    """Test GeometricMoGPriorConfig default values."""
    with pytest.warns(UserWarning, match="artificial topology"):
        config = GeometricMoGPriorConfig()
    assert config.num_components == 10
    assert config.geometric_arrangement == "circle"
    assert config.geometric_radius == 2.0
    assert config.get_prior_type() == "geometric_mog"


def test_geometric_mog_prior_config_validation():
    """Test GeometricMoGPriorConfig validation."""
    with pytest.raises(ValueError, match="geometric_arrangement must be one of"):
        GeometricMoGPriorConfig(geometric_arrangement="invalid")

    with pytest.raises(ValueError, match="geometric_radius must be positive"):
        GeometricMoGPriorConfig(geometric_radius=0)


# ============================================================================
# ExperimentConfig Tests
# ============================================================================


def test_experiment_config_defaults():
    """Test ExperimentConfig with default sub-configs."""
    config = ExperimentConfig()
    assert isinstance(config.network, NetworkConfig)
    assert isinstance(config.training, TrainingConfig)
    assert isinstance(config.loss, LossConfig)
    assert isinstance(config.decoder, DecoderFeatures)
    assert isinstance(config.prior, StandardPriorConfig)
    assert config.get_prior_type() == "standard"


def test_experiment_config_custom():
    """Test ExperimentConfig with custom sub-configs."""
    config = ExperimentConfig(
        network=NetworkConfig(latent_dim=10),
        training=TrainingConfig(batch_size=256),
        loss=LossConfig(recon_weight=1000.0),
        prior=MixturePriorConfig(num_components=20),
    )
    assert config.network.latent_dim == 10
    assert config.training.batch_size == 256
    assert config.loss.recon_weight == 1000.0
    assert config.prior.num_components == 20
    assert config.is_mixture_based_prior() is True


def test_experiment_config_component_embedding_dim_default():
    """Test that component_embedding_dim defaults to latent_dim."""
    config = ExperimentConfig(
        network=NetworkConfig(latent_dim=10),
        decoder=DecoderFeatures(component_embedding_dim=None),
    )
    assert config.decoder.component_embedding_dim == 10


def test_experiment_config_top_m_gating_validation():
    """Test top_m_gating validation against num_components."""
    with pytest.raises(ValueError, match="top_m_gating.*cannot exceed"):
        ExperimentConfig(
            prior=MixturePriorConfig(num_components=10),
            decoder=DecoderFeatures(top_m_gating=15),
        )


def test_experiment_config_tau_classifier_validation():
    """Test τ-classifier validation."""
    # Should pass: num_components >= num_classes
    config = ExperimentConfig(
        network=NetworkConfig(num_classes=10),
        prior=MixturePriorConfig(num_components=10, use_tau_classifier=True),
    )
    assert config.prior.use_tau_classifier is True

    # Should fail: num_components < num_classes
    with pytest.raises(ValueError, match="num_components must be >= num_classes"):
        ExperimentConfig(
            network=NetworkConfig(num_classes=10),
            prior=MixturePriorConfig(num_components=5, use_tau_classifier=True),
        )


def test_experiment_config_component_aware_decoder_warning():
    """Test warning for component-aware decoder with non-mixture prior."""
    with pytest.warns(UserWarning, match="use_component_aware_decoder.*only applies"):
        config = ExperimentConfig(
            prior=StandardPriorConfig(),
            decoder=DecoderFeatures(use_component_aware_decoder=True),
        )


# ============================================================================
# Converter Tests: dict → ExperimentConfig
# ============================================================================


def test_experiment_config_from_dict_modular_structure():
    """Test loading ExperimentConfig from modular dict."""
    config_dict = {
        "network": {
            "latent_dim": 10,
            "encoder_type": "conv",
        },
        "training": {
            "batch_size": 256,
        },
        "loss": {
            "recon_weight": 1000.0,
        },
        "prior": {
            "type": "mixture",
            "num_components": 20,
        },
    }

    config = experiment_config_from_dict(config_dict)
    assert config.network.latent_dim == 10
    assert config.network.encoder_type == "conv"
    assert config.training.batch_size == 256
    assert config.loss.recon_weight == 1000.0
    assert isinstance(config.prior, MixturePriorConfig)
    assert config.prior.num_components == 20


def test_experiment_config_from_dict_flat_structure():
    """Test loading ExperimentConfig from flat dict (legacy style)."""
    config_dict = {
        "latent_dim": 10,
        "encoder_type": "conv",
        "batch_size": 256,
        "recon_weight": 1000.0,
        "prior_type": "mixture",
        "num_components": 20,
    }

    config = experiment_config_from_dict(config_dict)
    assert config.network.latent_dim == 10
    assert config.network.encoder_type == "conv"
    assert config.training.batch_size == 256
    assert config.loss.recon_weight == 1000.0
    assert isinstance(config.prior, MixturePriorConfig)
    assert config.prior.num_components == 20


def test_experiment_config_from_dict_with_tuple_conversion():
    """Test that lists are converted to tuples for hidden_dims and input_hw."""
    config_dict = {
        "hidden_dims": [512, 256, 128],
        "input_hw": [28, 28],
    }

    config = experiment_config_from_dict(config_dict)
    assert config.network.hidden_dims == (512, 256, 128)
    assert config.network.input_hw == (28, 28)


def test_experiment_config_from_dict_all_prior_types():
    """Test loading all prior types from dict."""
    # Standard
    config = experiment_config_from_dict({"prior_type": "standard"})
    assert isinstance(config.prior, StandardPriorConfig)

    # Mixture
    config = experiment_config_from_dict({"prior_type": "mixture"})
    assert isinstance(config.prior, MixturePriorConfig)

    # Vamp
    config = experiment_config_from_dict({"prior_type": "vamp"})
    assert isinstance(config.prior, VampPriorConfig)

    # Geometric MoG
    with pytest.warns(UserWarning, match="artificial topology"):
        config = experiment_config_from_dict({"prior_type": "geometric_mog"})
    assert isinstance(config.prior, GeometricMoGPriorConfig)

    # Unknown
    with pytest.raises(ValueError, match="Unknown prior_type"):
        experiment_config_from_dict({"prior_type": "invalid"})


# ============================================================================
# Converter Tests: ExperimentConfig ↔ SSVAEConfig
# ============================================================================


