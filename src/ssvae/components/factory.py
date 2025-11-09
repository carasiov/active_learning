from __future__ import annotations

from typing import Tuple

from flax import linen as nn

from ssvae.config import SSVAEConfig
from .classifier import Classifier
from .tau_classifier import TauClassifier
from .decoders import (
    ComponentAwareConvDecoder,
    ComponentAwareDenseDecoder,
    ConvDecoder,
    DenseDecoder,
)
from .encoders import ConvEncoder, DenseEncoder, MixtureDenseEncoder


def _resolve_input_hw(config: SSVAEConfig, input_hw: Tuple[int, int] | None) -> Tuple[int, int]:
    if input_hw is not None:
        return input_hw
    if config.input_hw is not None:
        return config.input_hw
    raise ValueError("input_hw must be provided either via argument or SSVAEConfig.input_hw")


def _resolve_encoder_hidden_dims(config: SSVAEConfig, input_hw: Tuple[int, int]) -> Tuple[int, ...]:
    if config.hidden_dims:
        return config.hidden_dims
    flat = input_hw[0] * input_hw[1]
    return (flat,)


def build_encoder(config: SSVAEConfig, *, input_hw: Tuple[int, int] | None = None) -> nn.Module:
    resolved_hw = _resolve_input_hw(config, input_hw)
    
    if config.prior_type == "mixture":
        if config.encoder_type != "dense":
            raise ValueError(f"Mixture prior not supported with {config.encoder_type} encoder")
        hidden_dims = _resolve_encoder_hidden_dims(config, resolved_hw)
        return MixtureDenseEncoder(
            hidden_dims=hidden_dims,
            latent_dim=config.latent_dim,
            num_components=config.num_components,
        )
    
    if config.encoder_type == "dense":
        hidden_dims = _resolve_encoder_hidden_dims(config, resolved_hw)
        return DenseEncoder(hidden_dims=hidden_dims, latent_dim=config.latent_dim)
    if config.encoder_type == "conv":
        return ConvEncoder(latent_dim=config.latent_dim)
    raise ValueError(f"Unknown encoder type: {config.encoder_type}")


def build_decoder(config: SSVAEConfig, *, input_hw: Tuple[int, int] | None = None) -> nn.Module:
    """Build decoder based on configuration.

    For mixture prior with component-aware decoder enabled, creates decoders that
    process z and component embeddings separately for functional specialization.
    """
    resolved_hw = _resolve_input_hw(config, input_hw)

    # Component-aware decoders for mixture prior
    use_component_aware = (
        config.prior_type == "mixture" and
        config.use_component_aware_decoder
    )

    if config.decoder_type == "dense":
        hidden_dims = _resolve_encoder_hidden_dims(config, resolved_hw)
        decoder_hidden_dims = tuple(reversed(hidden_dims)) or (config.latent_dim,)

        if use_component_aware:
            return ComponentAwareDenseDecoder(
                hidden_dims=decoder_hidden_dims,
                output_hw=resolved_hw,
                component_embedding_dim=config.component_embedding_dim,
                latent_dim=config.latent_dim,
            )
        else:
            return DenseDecoder(hidden_dims=decoder_hidden_dims, output_hw=resolved_hw)

    if config.decoder_type == "conv":
        if use_component_aware:
            return ComponentAwareConvDecoder(
                latent_dim=config.latent_dim,
                output_hw=resolved_hw,
                component_embedding_dim=config.component_embedding_dim,
            )
        else:
            return ConvDecoder(latent_dim=config.latent_dim, output_hw=resolved_hw)

    raise ValueError(f"Unknown decoder type: {config.decoder_type}")


def build_classifier(config: SSVAEConfig, *, input_hw: Tuple[int, int] | None = None) -> nn.Module:
    """Build classifier based on configuration.

    Returns TauClassifier for mixture prior when use_tau_classifier=True,
    otherwise returns standard z-based Classifier.
    """
    # τ-based classifier for mixture prior
    if config.use_tau_classifier:
        if config.prior_type != "mixture":
            raise ValueError("τ-classifier requires mixture prior")
        return TauClassifier(
            num_components=config.num_components,
            num_classes=config.num_classes,
            alpha_0=config.tau_alpha_0,
        )

    # Standard z-based classifier
    if config.classifier_type == "dense":
        resolved_hw = _resolve_input_hw(config, input_hw)
        hidden_dims = _resolve_encoder_hidden_dims(config, resolved_hw)
        last_hidden = hidden_dims[-1] if hidden_dims else config.latent_dim
        classifier_hidden_dims = (last_hidden, last_hidden)
        return Classifier(
            hidden_dims=classifier_hidden_dims,
            num_classes=config.num_classes,
            dropout_rate=config.dropout_rate,
        )
    if config.classifier_type == "conv":
        raise NotImplementedError("ConvClassifier not yet implemented")
    raise ValueError(f"Unknown classifier type: {config.classifier_type}")


def get_architecture_dims(config: SSVAEConfig, *, input_hw: Tuple[int, int]) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    encoder = build_encoder(config, input_hw=input_hw)
    decoder = build_decoder(config, input_hw=input_hw)
    classifier = build_classifier(config, input_hw=input_hw)
    encoder_dims = getattr(encoder, "hidden_dims", ())
    decoder_dims = getattr(decoder, "hidden_dims", ())
    classifier_dims = getattr(classifier, "hidden_dims", ())
    return encoder_dims, decoder_dims, classifier_dims
