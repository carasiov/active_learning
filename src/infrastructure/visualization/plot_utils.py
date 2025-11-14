"""Shared utilities for visualization module.

This module provides common helper functions and classes used across
all plotting functions in the visualization package.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def _sanitize_model_name(name: str) -> str:
    """Sanitize model name for use in filenames.

    Converts a model name into a safe filename by replacing non-alphanumeric
    characters with underscores.

    Args:
        name: Model name to sanitize

    Returns:
        Sanitized filename-safe string

    Example:
        >>> _sanitize_model_name("My-Model (v2)")
        'my_model_v2'
    """
    return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_") or "model"


def _build_label_palette(num_labels: int) -> np.ndarray:
    """Build a stable color palette for class labels.

    Returns an RGBA palette with colors that remain consistent across
    different numbers of classes.

    Args:
        num_labels: Number of distinct class labels

    Returns:
        RGBA array of shape (num_labels, 4) with opaque colors
    """
    if num_labels <= 10:
        base_cmap = cm.get_cmap("tab10")
    elif num_labels <= 20:
        base_cmap = cm.get_cmap("tab20")
    else:
        base_cmap = cm.get_cmap("nipy_spectral")

    anchors = np.linspace(0, 1, num_labels, endpoint=False) if num_labels > 0 else np.array([0.0])
    palette = np.stack([base_cmap(anchor) for anchor in anchors], axis=0) if num_labels > 0 else np.ones((1, 4))
    palette[:, 3] = 1.0  # Ensure opaque baseline colors
    return palette


def _downsample_points(
    latent: np.ndarray,
    responsibilities: np.ndarray,
    labels: np.ndarray,
    *,
    max_points: int = 20000,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subsample points for plotting to keep figures readable.

    When visualizing large datasets, downsampling prevents overcrowding
    and reduces rendering time.

    Args:
        latent: Latent space coordinates (N, D)
        responsibilities: Component responsibilities (N, K)
        labels: Class labels (N,)
        max_points: Maximum number of points to retain
        seed: Random seed for reproducibility

    Returns:
        Tuple of (downsampled_latent, downsampled_responsibilities, downsampled_labels)
    """
    total = latent.shape[0]
    if total <= max_points:
        return latent, responsibilities, labels

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(total, size=max_points, replace=False))
    return latent[indices], responsibilities[indices], labels[indices]


def _compute_limits(
    latent: np.ndarray,
    *,
    padding: float = 0.05
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute shared axis limits with padding for consistent layouts.

    Ensures all subplots in a grid share the same axis ranges for easy comparison.

    Args:
        latent: Latent space coordinates (N, 2)
        padding: Fraction of range to add as padding

    Returns:
        Tuple of ((x_min, x_max), (y_min, y_max))
    """
    x_min, x_max = latent[:, 0].min(), latent[:, 0].max()
    y_min, y_max = latent[:, 1].min(), latent[:, 1].max()
    x_pad = (x_max - x_min) * padding or 1e-3
    y_pad = (y_max - y_min) * padding or 1e-3
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def _extract_component_recon(
    extras: Dict[str, Any]
) -> Tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """Extract per-component reconstructions from model extras.

    Component-aware decoders return reconstructions for each mixture component.
    This function extracts and validates that data.

    Args:
        extras: Dictionary of extra outputs from model forward pass

    Returns:
        Tuple of (recon_per_component, sigma_per_component, error_message).
        If successful, error_message is None. Otherwise, first two elements are None.
    """
    recon_per_component = extras.get("recon_per_component")
    if recon_per_component is None:
        return None, None, "recon_per_component not found in extras"

    sigma_per_component = None
    if isinstance(recon_per_component, tuple):
        recon_per_component, sigma_per_component = recon_per_component

    recon_per_component = np.asarray(recon_per_component)
    if recon_per_component.ndim < 4:
        return None, None, f"unexpected recon_per_component shape {recon_per_component.shape}"
    if recon_per_component.shape[0] != 1:
        return None, None, f"expected single-sample recon, got batch {recon_per_component.shape[0]}"
    recon_per_component = recon_per_component[0]

    if sigma_per_component is not None:
        sigma_per_component = np.asarray(sigma_per_component)
        if sigma_per_component.ndim >= 2:
            sigma_per_component = sigma_per_component[0]

    return recon_per_component, sigma_per_component, None


def _prep_image(img: np.ndarray) -> np.ndarray:
    """Prepare image array for matplotlib visualization.

    Removes singleton channel dimension if present to ensure grayscale
    images display correctly.

    Args:
        img: Image array with shape (H, W) or (H, W, 1) or (H, W, C)

    Returns:
        Image array suitable for plt.imshow()
    """
    if img.ndim == 3 and img.shape[-1] == 1:
        return img[..., 0]
    return img


class PlotGrid:
    """Helper class for creating subplot grids with consistent layouts.

    Simplifies the common pattern of creating multi-panel figures where
    the number of subplots depends on the number of models or components.

    Example:
        >>> grid = PlotGrid(num_items=5, max_cols=3, subplot_size=(6, 5))
        >>> fig, axes = grid.create()
        >>> for ax, item in zip(grid.iterate(axes), items):
        >>>     ax.plot(item)
        >>> grid.hide_unused(axes)
    """

    def __init__(
        self,
        num_items: int,
        max_cols: int = 3,
        subplot_size: Tuple[float, float] = (6, 5)
    ):
        """Initialize plot grid configuration.

        Args:
            num_items: Number of subplots needed
            max_cols: Maximum number of columns in grid
            subplot_size: Size of each subplot (width, height) in inches
        """
        self.num_items = num_items
        self.n_cols = min(max_cols, num_items)
        self.n_rows = (num_items + self.n_cols - 1) // self.n_cols
        self.subplot_size = subplot_size

    def create(self) -> Tuple[plt.Figure, np.ndarray | plt.Axes]:
        """Create matplotlib figure with subplot grid.

        Returns:
            Tuple of (figure, axes) where axes is either an ndarray or single Axes
        """
        figsize = (
            self.subplot_size[0] * self.n_cols,
            self.subplot_size[1] * self.n_rows
        )
        fig, axes = plt.subplots(self.n_rows, self.n_cols, figsize=figsize)
        return fig, axes

    def iterate(self, axes: np.ndarray | plt.Axes):
        """Iterate over axes in consistent manner regardless of grid shape.

        Args:
            axes: Axes array or single Axes from create()

        Yields:
            Individual Axes objects
        """
        if self.num_items == 1:
            yield axes
        else:
            axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
            for ax in axes_flat[:self.num_items]:
                yield ax

    def hide_unused(self, axes: np.ndarray | plt.Axes):
        """Hide any unused subplots in the grid.

        Args:
            axes: Axes array or single Axes from create()
        """
        if self.num_items == 1 or not isinstance(axes, np.ndarray):
            return

        axes_flat = axes.flatten()
        for idx in range(self.num_items, len(axes_flat)):
            axes_flat[idx].set_visible(False)


def safe_save_plot(
    fig: plt.Figure,
    output_path: Path,
    dpi: int = 150,
    verbose: bool = True
) -> bool:
    """Safely save a matplotlib figure with error handling.

    Args:
        fig: Matplotlib figure to save
        output_path: Path where figure should be saved
        dpi: Resolution in dots per inch
        verbose: Whether to print success message

    Returns:
        True if save succeeded, False otherwise
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        if verbose:
            print(f"  Saved: {output_path}")
        plt.close(fig)
        return True
    except Exception as e:
        print(f"  Error saving {output_path}: {e}")
        plt.close(fig)
        return False
