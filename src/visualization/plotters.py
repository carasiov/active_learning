"""Registry bindings for visualization pipeline.

This module serves as the entry point for the visualization system, registering
all plotting functions with the visualization registry. The actual implementation
is split across domain-specific modules:

- core_plots: Basic VAE diagnostics (loss, latent space, reconstructions)
- mixture_plots: Mixture prior specific visualizations
- tau_plots: τ-classifier semi-supervised learning diagnostics
- plot_utils: Shared utility functions

The registry pattern allows dynamic discovery and execution of visualization
components based on model configuration.
"""
from __future__ import annotations

from .registry import VisualizationContext, register_plotter
from common.status import ComponentResult

# Import plotting functions from domain modules
from .core_plots import (
    plot_loss_comparison,
    plot_latent_spaces,
    plot_reconstructions,
    generate_report,
)
from .mixture_plots import (
    plot_latent_by_component,
    plot_channel_latent_responsibility,
    plot_responsibility_histogram,
    plot_mixture_evolution,
    plot_component_embedding_divergence,
    plot_reconstruction_by_component,
)
from .tau_plots import (
    plot_tau_matrix_heatmap,
    plot_tau_per_class_accuracy,
    plot_tau_certainty_analysis,
)


# ---------------------------------------------------------------------------
# Helper functions for backward compatibility
# ---------------------------------------------------------------------------

def _single_model_dict(model):
    """Wrap single model in dictionary for functions expecting Dict[str, model]."""
    return {"Model": model}


def _single_history_dict(history):
    """Wrap single history in dictionary for functions expecting Dict[str, history]."""
    return {"Model": history}


# ---------------------------------------------------------------------------
# Registry bindings - Core visualizations
# ---------------------------------------------------------------------------

@register_plotter
def loss_curves_plotter(context: VisualizationContext) -> ComponentResult:
    """Generate loss curves plot.

    Always runs - provides fundamental training diagnostic information.

    Returns:
        ComponentResult with success/failure status
    """
    try:
        plot_loss_comparison(_single_history_dict(context.history), context.figures_dir)
        return ComponentResult.success(data={})
    except Exception as e:
        return ComponentResult.failed(
            reason="Failed to generate loss curves",
            error=e,
        )


@register_plotter
def latent_space_plotter(context: VisualizationContext) -> ComponentResult:
    """Generate latent space visualization.

    Always runs - shows learned latent representations colored by class.

    Returns:
        ComponentResult with success/failure status
    """
    try:
        plot_latent_spaces(_single_model_dict(context.model), context.x_train, context.y_true, context.figures_dir)
        return ComponentResult.success(data={})
    except Exception as e:
        return ComponentResult.failed(
            reason="Failed to generate latent space plot",
            error=e,
        )


@register_plotter
def reconstructions_plotter(context: VisualizationContext) -> ComponentResult:
    """Generate reconstruction visualizations.

    Always runs - assesses model's ability to reconstruct inputs.

    Returns:
        ComponentResult with reconstruction paths in data dict
    """
    try:
        paths = plot_reconstructions(_single_model_dict(context.model), context.x_train, context.figures_dir)
        return ComponentResult.success(data={"reconstructions": paths})
    except Exception as e:
        return ComponentResult.failed(
            reason="Failed to generate reconstructions",
            error=e,
        )


# ---------------------------------------------------------------------------
# Registry bindings - Mixture prior visualizations
# ---------------------------------------------------------------------------

@register_plotter
def latent_by_component_plotter(context: VisualizationContext) -> ComponentResult:
    """Generate latent space colored by component assignment.

    Only applicable for mixture priors - shows component specialization.

    Returns:
        ComponentResult.disabled if not mixture prior, otherwise success/failed
    """
    if getattr(context.config, "prior_type", "standard") != "mixture":
        return ComponentResult.disabled(
            reason="Requires mixture prior"
        )

    try:
        plot_latent_by_component(_single_model_dict(context.model), context.x_train, context.y_true, context.figures_dir)
        return ComponentResult.success(data={})
    except Exception as e:
        return ComponentResult.failed(
            reason="Failed to generate latent by component plot",
            error=e,
        )


@register_plotter
def channel_latent_responsibility_plotter(context: VisualizationContext) -> ComponentResult:
    """Visualize channel-wise latent usage with label colors and responsibility-based alpha.

    Only applicable for mixture priors with 2D latent space - provides detailed
    per-component analysis of latent space structure.

    Returns:
        ComponentResult with appropriate status
    """
    if getattr(context.config, "prior_type", "standard") != "mixture":
        return ComponentResult.disabled(reason="Requires mixture prior")

    if getattr(context.config, "latent_dim", None) != 2:
        return ComponentResult.skipped(reason="Requires 2D latent space")

    try:
        paths = plot_channel_latent_responsibility(
            _single_model_dict(context.model),
            context.x_train,
            context.y_true,
            context.figures_dir,
        )
        if not paths:
            return ComponentResult.skipped(reason="Responsibilities unavailable for channel latent plots")
        return ComponentResult.success(data={"channel_latents": paths})
    except Exception as e:
        return ComponentResult.failed(
            reason="Failed to generate channel latent plots",
            error=e,
        )


@register_plotter
def responsibility_histogram_plotter(context: VisualizationContext) -> ComponentResult:
    """Generate responsibility confidence histogram.

    Only applicable for mixture priors - shows distribution of max_c q(c|x).

    Returns:
        ComponentResult with appropriate status
    """
    if getattr(context.config, "prior_type", "standard") != "mixture":
        return ComponentResult.disabled(
            reason="Requires mixture prior"
        )

    try:
        plot_responsibility_histogram(_single_model_dict(context.model), context.figures_dir)
        return ComponentResult.success(data={})
    except Exception as e:
        return ComponentResult.failed(
            reason="Failed to generate responsibility histogram",
            error=e,
        )


@register_plotter
def mixture_evolution_plotter(context: VisualizationContext) -> ComponentResult:
    """Generate π and component usage evolution plots.

    Only applicable for mixture priors - tracks mixture dynamics during training.

    Returns:
        ComponentResult with appropriate status
    """
    if getattr(context.config, "prior_type", "standard") != "mixture":
        return ComponentResult.disabled(
            reason="Requires mixture prior"
        )

    try:
        plot_mixture_evolution(_single_model_dict(context.model), context.figures_dir)
        return ComponentResult.success(data={})
    except Exception as e:
        return ComponentResult.failed(
            reason="Failed to generate mixture evolution plots",
            error=e,
        )


@register_plotter
def component_embedding_plotter(context: VisualizationContext) -> ComponentResult:
    """Generate component embedding and reconstruction-by-component visualizations.

    Only applicable for mixture priors with component-aware decoders - shows
    whether components learn distinct reconstruction strategies.

    Returns:
        ComponentResult with appropriate status
    """
    if getattr(context.config, "prior_type", "standard") != "mixture":
        return ComponentResult.disabled(
            reason="Requires mixture prior"
        )

    if not getattr(context.config, "use_component_aware_decoder", False):
        return ComponentResult.disabled(
            reason="Requires component-aware decoder"
        )

    try:
        plot_component_embedding_divergence(_single_model_dict(context.model), context.figures_dir)
        plot_reconstruction_by_component(_single_model_dict(context.model), context.x_train, context.figures_dir)
        return ComponentResult.success(data={})
    except Exception as e:
        return ComponentResult.failed(
            reason="Failed to generate component embedding plots",
            error=e,
        )


# ---------------------------------------------------------------------------
# Registry bindings - τ-classifier visualizations
# ---------------------------------------------------------------------------

@register_plotter
def tau_matrix_plotter(context: VisualizationContext) -> ComponentResult:
    """Generate τ-classifier visualization suite.

    Includes τ matrix heatmap, per-class accuracy, and certainty analysis.
    Only applicable when τ-classifier is enabled.

    Returns:
        ComponentResult with appropriate status
    """
    if getattr(context.model, "_tau_classifier", None) is None:
        return ComponentResult.disabled(
            reason="τ-classifier not available"
        )

    try:
        plot_tau_matrix_heatmap(_single_model_dict(context.model), context.figures_dir)
        plot_tau_per_class_accuracy(_single_model_dict(context.model), context.x_train, context.y_true, context.figures_dir)
        plot_tau_certainty_analysis(_single_model_dict(context.model), context.x_train, context.y_true, context.figures_dir)
        return ComponentResult.success(data={})
    except Exception as e:
        return ComponentResult.failed(
            reason="Failed to generate τ-classifier plots",
            error=e,
        )
