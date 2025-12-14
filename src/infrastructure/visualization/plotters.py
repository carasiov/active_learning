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
from infrastructure import ComponentResult

# Import plotting functions from domain modules
from .core import (
    plot_loss_comparison,
    plot_latent_spaces,
    plot_reconstructions,
    generate_report,
)
from .mixture import (
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
from .tau import (
    plot_tau_matrix_heatmap,
    plot_tau_per_class_accuracy,
    plot_tau_certainty_analysis,
)
from .curriculum import plot_curriculum_metrics, plot_curriculum_channel_progression


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
    if not context.config.is_mixture_based_prior():
        return ComponentResult.disabled(
            reason="Requires mixture-based prior"
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
    if not context.config.is_mixture_based_prior():
        return ComponentResult.disabled(reason="Requires mixture-based prior")

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
    if not context.config.is_mixture_based_prior():
        return ComponentResult.disabled(
            reason="Requires mixture-based prior"
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
    if not context.config.is_mixture_based_prior():
        return ComponentResult.disabled(
            reason="Requires mixture-based prior"
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
    if not context.config.is_mixture_based_prior():
        return ComponentResult.disabled(
            reason="Requires mixture-based prior"
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


@register_plotter
def selected_vs_weighted_recon_plotter(context: VisualizationContext) -> ComponentResult:
    """Visualize selected-channel vs weighted reconstructions (hard vs soft routing)."""
    if not context.config.is_mixture_based_prior():
        return ComponentResult.disabled(reason="Requires mixture-based prior")

    try:
        plot_selected_vs_weighted_reconstruction(
            _single_model_dict(context.model),
            context.x_train,
            context.figures_dir,
        )
        return ComponentResult.success(data={})
    except Exception as e:
        return ComponentResult.failed(
            reason="Failed to generate selected vs weighted reconstructions",
            error=e,
        )


@register_plotter
def channel_ownership_plotter(context: VisualizationContext) -> ComponentResult:
    """Heatmap of channel vs label responsibility (specialization view)."""
    if not context.config.is_mixture_based_prior():
        return ComponentResult.disabled(reason="Requires mixture-based prior")

    try:
        plot_channel_ownership_heatmap(
            _single_model_dict(context.model),
            context.x_train,
            context.y_true,
            context.figures_dir,
        )
        return ComponentResult.success(data={})
    except Exception as e:
        return ComponentResult.failed(
            reason="Failed to generate channel ownership heatmap",
            error=e,
        )


@register_plotter
def component_kl_heatmap_plotter(context: VisualizationContext) -> ComponentResult:
    """Per-component KL heatmap to spot dead/over-regularized channels."""
    if not context.config.is_mixture_based_prior():
        return ComponentResult.disabled(reason="Requires mixture-based prior")

    try:
        plot_component_kl_heatmap(
            _single_model_dict(context.model),
            context.x_train,
            context.figures_dir,
        )
        return ComponentResult.success(data={})
    except Exception as e:
        return ComponentResult.failed(
            reason="Failed to generate component KL heatmap",
            error=e,
        )


@register_plotter
def routing_hardness_plotter(context: VisualizationContext) -> ComponentResult:
    """Compare routing hardness with and without Gumbel sampling."""
    if not context.config.is_mixture_based_prior():
        return ComponentResult.disabled(reason="Requires mixture-based prior")

    try:
        plot_routing_hardness(
            _single_model_dict(context.model),
            context.x_train,
            context.figures_dir,
        )
        return ComponentResult.success(data={})
    except Exception as e:
        return ComponentResult.failed(
            reason="Failed to generate routing hardness plot",
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


# ---------------------------------------------------------------------------
# Registry bindings - Curriculum visualizations
# ---------------------------------------------------------------------------

@register_plotter
def curriculum_metrics_plotter(context: VisualizationContext) -> ComponentResult:
    """Generate curriculum learning diagnostic plots.

    Shows k_active evolution, migration window indicators, and trigger-based
    unlock diagnostics (normality score, plateau detection).

    Only applicable when curriculum learning is enabled.

    Returns:
        ComponentResult with appropriate status
    """
    # Check if curriculum is enabled
    if not getattr(context.config, "curriculum_enabled", False):
        return ComponentResult.disabled(
            reason="Curriculum learning not enabled"
        )

    # Check if history has curriculum data
    if "k_active" not in context.history or len(context.history.get("k_active", [])) == 0:
        return ComponentResult.skipped(
            reason="No curriculum metrics in training history"
        )

    try:
        success = plot_curriculum_metrics(
            _single_history_dict(context.history),
            context.figures_dir,
        )
        if success:
            return ComponentResult.success(data={})
        else:
            return ComponentResult.skipped(
                reason="No curriculum data to plot"
            )
    except Exception as e:
        return ComponentResult.failed(
            reason="Failed to generate curriculum metrics plot",
            error=e,
        )


@register_plotter
def curriculum_channel_progression_plotter(context: VisualizationContext) -> ComponentResult:
    """Generate curriculum-aware channel latent visualization.

    Shows channels in unlock order with annotations about when each was unlocked
    and usage statistics. Provides a curriculum-centric view of channel learning.

    Only applicable when curriculum learning is enabled with mixture priors.

    Returns:
        ComponentResult with appropriate status
    """
    # Check if curriculum is enabled
    if not getattr(context.config, "curriculum_enabled", False):
        return ComponentResult.disabled(
            reason="Curriculum learning not enabled"
        )

    # Check if mixture-based prior
    if not context.config.is_mixture_based_prior():
        return ComponentResult.disabled(
            reason="Requires mixture-based prior"
        )

    # Check if history has curriculum data
    if "k_active" not in context.history or len(context.history.get("k_active", [])) == 0:
        return ComponentResult.skipped(
            reason="No curriculum metrics in training history"
        )

    try:
        path = plot_curriculum_channel_progression(
            context.model,
            context.x_train,
            context.y_true,
            context.history,
            context.figures_dir,
        )
        if path:
            return ComponentResult.success(data={"channel_progression": path})
        else:
            return ComponentResult.skipped(
                reason="Could not generate curriculum channel progression"
            )
    except Exception as e:
        import traceback
        print(f"curriculum_channel_progression error: {e}")
        traceback.print_exc()
        return ComponentResult.failed(
            reason="Failed to generate curriculum channel progression plot",
            error=e,
        )


@register_plotter
def curriculum_evolution_plotter(context: VisualizationContext) -> ComponentResult:
    """Generate curriculum evolution visualization showing temporal progression.

    Shows a grid of channels (rows) × time stages (columns), displaying how each
    channel's latent space evolved from unlock through migration end.

    Requires curriculum snapshots to have been saved during training.

    Returns:
        ComponentResult with appropriate status
    """
    from .curriculum import plot_curriculum_evolution
    from rcmvae.application.services.diagnostics_service import DiagnosticsCollector

    # Check if curriculum is enabled
    if not getattr(context.config, "curriculum_enabled", False):
        return ComponentResult.disabled(
            reason="Curriculum learning not enabled"
        )

    # Check if mixture-based prior
    if not context.config.is_mixture_based_prior():
        return ComponentResult.disabled(
            reason="Requires mixture-based prior"
        )

    # Load curriculum snapshots - check multiple possible locations
    try:
        snapshots = []
        search_paths = [
            context.figures_dir.parent,
            context.figures_dir.parent / "artifacts",  # snapshots saved here
            context.figures_dir.parent / "checkpoints",
            context.figures_dir,
        ]
        print(f"[curriculum_evolution] Looking for snapshots in: {[str(p) for p in search_paths]}")

        for search_path in search_paths:
            if search_path.exists():
                snapshots = DiagnosticsCollector.load_curriculum_snapshots(search_path)
                if snapshots:
                    print(f"[curriculum_evolution] Found {len(snapshots)} snapshots in {search_path}")
                    break

        if not snapshots:
            print(f"[curriculum_evolution] No snapshots found in any location")
            return ComponentResult.skipped(
                reason="No curriculum snapshots found (run with snapshot saving enabled)"
            )
    except Exception as e:
        print(f"[curriculum_evolution] Error loading snapshots: {e}")
        return ComponentResult.skipped(
            reason=f"Could not load curriculum snapshots: {e}"
        )

    try:
        path = plot_curriculum_evolution(
            snapshots,
            context.figures_dir,
            num_classes=context.config.num_classes,
        )
        if path:
            return ComponentResult.success(data={"channel_evolution": path})
        else:
            return ComponentResult.skipped(
                reason="Could not generate curriculum evolution plot"
            )
    except Exception as e:
        import traceback
        print(f"curriculum_evolution error: {e}")
        traceback.print_exc()
        return ComponentResult.failed(
            reason="Failed to generate curriculum evolution plot",
            error=e,
        )
