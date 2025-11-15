"""Mixture prior visualization components."""

from .plots import (
    plot_latent_by_component,
    plot_channel_latent_responsibility,
    plot_responsibility_histogram,
    plot_mixture_evolution,
    plot_component_embedding_divergence,
    plot_reconstruction_by_component,
)

__all__ = [
    "plot_latent_by_component",
    "plot_channel_latent_responsibility",
    "plot_responsibility_histogram",
    "plot_mixture_evolution",
    "plot_component_embedding_divergence",
    "plot_reconstruction_by_component",
]
