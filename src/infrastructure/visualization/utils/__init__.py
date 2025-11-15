"""Utility subpackage for visualization helpers."""

from ._plot_utils import (
    _sanitize_model_name,
    _build_label_palette,
    _downsample_points,
    _compute_limits,
    _extract_component_recon,
    _prep_image,
    PlotGrid,
    safe_save_plot,
    style_axes,
    LATENT_POINT_SIZE,
    CHANNEL_POINT_SIZE,
)

__all__ = [
    "_sanitize_model_name",
    "_build_label_palette",
    "_downsample_points",
    "_compute_limits",
    "_extract_component_recon",
    "_prep_image",
    "PlotGrid",
    "safe_save_plot",
    "style_axes",
    "LATENT_POINT_SIZE",
    "CHANNEL_POINT_SIZE",
]
