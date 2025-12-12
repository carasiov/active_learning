"""Mixture prior visualization components."""

from .plots import (
    plot_latent_by_component,
    plot_channel_latent_responsibility,
    plot_responsibility_histogram,
    plot_mixture_evolution,
    plot_component_embedding_divergence,
    plot_reconstruction_by_component,
    plot_selected_vs_weighted_reconstruction,
    plot_channel_ownership_heatmap,
    plot_component_kl_heatmap,
    plot_routing_hardness,
)

__all__ = [
    "plot_latent_by_component",
    "plot_channel_latent_responsibility",
    "plot_responsibility_histogram",
    "plot_mixture_evolution",
    "plot_component_embedding_divergence",
    "plot_reconstruction_by_component",
    "plot_selected_vs_weighted_reconstruction",
    "plot_channel_ownership_heatmap",
    "plot_component_kl_heatmap",
    "plot_routing_hardness",
]
