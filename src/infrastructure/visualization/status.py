"""Status objects for robust visualization rendering.

This module provides the same ComponentResult pattern as metrics but for
visualization/plotting functions. See src/metrics/status.py for detailed
design rationale.

Usage:
    from .status import ComponentResult

    @register_plotter
    def plot_mixture_evolution(context: VisualizationContext) -> ComponentResult:
        if not context.config.is_mixture_based_prior():
            return ComponentResult.disabled(
                reason="Requires mixture-based prior"
            )

        mixture_dir = context.figures_dir / "mixture"
        mixture_dir.mkdir(exist_ok=True)

        try:
            fig = create_plot(context)
            path = mixture_dir / "evolution.png"
            fig.savefig(path)
            return ComponentResult.success(data={"path": str(path)})
        except Exception as e:
            return ComponentResult.failed(
                reason="Plot generation failed",
                error=e
            )
"""
from __future__ import annotations

# Import directly from common.status to ensure consistency
# Both metrics and visualizations use the same status model
from infrastructure import ComponentResult, ComponentStatus

__all__ = ["ComponentResult", "ComponentStatus"]
