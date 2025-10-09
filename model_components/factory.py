from __future__ import annotations

from typing import Tuple

from configs.base import SSVAEConfig
from .classifier import Classifier
from .decoders import DenseDecoder
from .encoders import DenseEncoder


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


def build_encoder(config: SSVAEConfig, *, input_hw: Tuple[int, int] | None = None) -> DenseEncoder:
    resolved_hw = _resolve_input_hw(config, input_hw)
    hidden_dims = _resolve_encoder_hidden_dims(config, resolved_hw)
    if config.encoder_type == "dense":
        return DenseEncoder(hidden_dims=hidden_dims, latent_dim=config.latent_dim)
    if config.encoder_type == "conv":
        raise NotImplementedError("ConvEncoder not yet implemented")
    raise ValueError(f"Unknown encoder type: {config.encoder_type}")


def build_decoder(config: SSVAEConfig, *, input_hw: Tuple[int, int] | None = None) -> DenseDecoder:
    resolved_hw = _resolve_input_hw(config, input_hw)
    hidden_dims = _resolve_encoder_hidden_dims(config, resolved_hw)
    decoder_hidden_dims = tuple(reversed(hidden_dims)) or (config.latent_dim,)
    if config.decoder_type == "dense":
        return DenseDecoder(hidden_dims=decoder_hidden_dims, output_hw=resolved_hw)
    if config.decoder_type == "conv":
        raise NotImplementedError("ConvDecoder not yet implemented")
    raise ValueError(f"Unknown decoder type: {config.decoder_type}")


def build_classifier(config: SSVAEConfig, *, input_hw: Tuple[int, int] | None = None) -> Classifier:
    if config.classifier_type == "dense":
        resolved_hw = _resolve_input_hw(config, input_hw)
        hidden_dims = _resolve_encoder_hidden_dims(config, resolved_hw)
        last_hidden = hidden_dims[-1] if hidden_dims else config.latent_dim
        classifier_hidden_dims = (last_hidden, last_hidden)
        return Classifier(hidden_dims=classifier_hidden_dims, num_classes=10)
    if config.classifier_type == "conv":
        raise NotImplementedError("ConvClassifier not yet implemented")
    raise ValueError(f"Unknown classifier type: {config.classifier_type}")


def get_architecture_dims(config: SSVAEConfig, *, input_hw: Tuple[int, int]) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    encoder = build_encoder(config, input_hw=input_hw)
    decoder = build_decoder(config, input_hw=input_hw)
    classifier = build_classifier(config, input_hw=input_hw)
    return encoder.hidden_dims, decoder.hidden_dims, classifier.hidden_dims

