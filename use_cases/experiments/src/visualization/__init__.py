"""Backward compatibility wrapper for visualization module.

IMPORTANT: The visualization module has been moved to src/visualization/ at the
repository root to enable reuse across projects (experiments, dashboard, etc.).

This module re-exports the new location for backward compatibility with existing
code that imports from use_cases/experiments/src/visualization/.

New code should import directly from:
    from visualization import VisualizationContext, render_all_plots

Old code continues to work:
    from ..visualization import VisualizationContext, render_all_plots
"""

# Re-export everything from the new location
from visualization import (
    VisualizationContext,
    register_plotter,
    render_all_plots,
)

__all__ = [
    "VisualizationContext",
    "register_plotter",
    "render_all_plots",
]
