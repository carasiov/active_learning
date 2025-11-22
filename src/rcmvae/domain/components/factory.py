from __future__ import annotations

from typing import Tuple

from flax import linen as nn

from rcmvae.domain.components.decoder_modules import (
    ConcatConditioner,
    ConvBackbone,
    DenseBackbone,
    FiLMLayer,
    HeteroscedasticHead,
    NoopConditioner,
    StandardHead,
)
from rcmvae.domain.config import SSVAEConfig
from .classifier import Classifier
from .decoders import ModularConvDecoder, ModularDenseDecoder
from .encoders import ConvEncoder, DenseEncoder, MixtureConvEncoder, MixtureDenseEncoder


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

    # All mixture-based priors (mixture, vamp, geometric_mog) use mixture encoder
    # which outputs component logits in addition to z_mean and z_log_var
    if config.is_mixture_based_prior():
        if config.encoder_type == "dense":
            hidden_dims = _resolve_encoder_hidden_dims(config, resolved_hw)
            return MixtureDenseEncoder(
                hidden_dims=hidden_dims,
                latent_dim=config.latent_dim,
                num_components=config.num_components,
                latent_layout=config.latent_layout,
            )
        if config.encoder_type == "conv":
            return MixtureConvEncoder(
                latent_dim=config.latent_dim,
                num_components=config.num_components,
                latent_layout=config.latent_layout,
            )
        raise ValueError(f"Mixture-based prior ({config.prior_type}) not supported with encoder_type '{config.encoder_type}'")

    if config.encoder_type == "dense":
        hidden_dims = _resolve_encoder_hidden_dims(config, resolved_hw)
        return DenseEncoder(hidden_dims=hidden_dims, latent_dim=config.latent_dim)
    if config.encoder_type == "conv":
        return ConvEncoder(latent_dim=config.latent_dim)
    raise ValueError(f"Unknown encoder type: {config.encoder_type}")


def build_decoder(config: SSVAEConfig, *, input_hw: Tuple[int, int] | None = None) -> nn.Module:
    """Build decoder using modular composition with no silent overrides."""
    resolved_hw = _resolve_input_hw(config, input_hw)

    use_film = config.prior_type in {"mixture", "geometric_mog"} and config.use_film_decoder
    use_concat = config.prior_type in {"mixture", "geometric_mog"} and config.use_component_aware_decoder
    use_heteroscedastic = config.use_heteroscedastic_decoder

    if use_film:
        conditioner: nn.Module = FiLMLayer(component_embedding_dim=config.component_embedding_dim)
    elif use_concat:
        conditioner = ConcatConditioner()
    else:
        conditioner = NoopConditioner()

    if config.decoder_type == "conv":
        backbone: nn.Module = ConvBackbone(latent_dim=config.latent_dim, output_hw=resolved_hw)
    elif config.decoder_type == "dense":
        hidden_dims = _resolve_encoder_hidden_dims(config, resolved_hw)
        decoder_hidden_dims = tuple(reversed(hidden_dims)) or (config.latent_dim,)
        backbone = DenseBackbone(hidden_dims=decoder_hidden_dims)
    else:
        raise ValueError(f"Unknown decoder type: {config.decoder_type}")

    if use_heteroscedastic:
        output_head: nn.Module = HeteroscedasticHead(
            output_hw=resolved_hw,
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
        )
    else:
        output_head = StandardHead(output_hw=resolved_hw)

    if config.decoder_type == "conv":
        return ModularConvDecoder(
            conditioner=conditioner,
            backbone=backbone,
            output_head=output_head,
        )
    return ModularDenseDecoder(
        conditioner=conditioner,
        backbone=backbone,
        output_head=output_head,
    )

    raise ValueError(f"Unknown decoder type: {config.decoder_type}")


def build_classifier(config: SSVAEConfig, *, input_hw: Tuple[int, int] | None = None) -> Classifier:
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
