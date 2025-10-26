"""Utility helpers for dashboard visualizations and rendering."""

from __future__ import annotations

import base64
import io
from typing import List

import numpy as np
from matplotlib import colormaps, colors as mcolors
from PIL import Image

# infoteam secondary color palette (colorblind-safe)
INFOTEAM_PALETTE = [
    '#C10A27',  # infoteam red (primary)
    '#45717A',  # teal
    '#AFCC37',  # lime green
    '#A69DCD',  # lavender
    '#8BBFC2',  # light teal
    '#F6C900',  # yellow
    '#4DBDC6',  # cyan
    '#554596',  # purple
    '#CFE09B',  # light green
    '#F6E3AC',  # cream
    '#6F6F6F'   # infoteam mid gray for unlabeled
]

VIRIDIS_CMAP = colormaps['viridis']


def _values_to_hex_viridis(values: np.ndarray) -> List[str]:
    """Map numeric values onto hex color codes using the viridis colormap (colorblind-safe)."""
    if values.size == 0:
        return []
    values = np.array(values, dtype=np.float64)
    v_min = float(np.nanmin(values))
    v_max = float(np.nanmax(values))
    if np.isclose(v_min, v_max):
        normed = np.zeros_like(values, dtype=np.float64)
    else:
        normed = (values - v_min) / (v_max - v_min)
        normed = np.clip(normed, 0.0, 1.0)
    rgba = VIRIDIS_CMAP(normed)
    return [mcolors.to_hex(color[:3]) for color in rgba]


def _colorize_user_labels(labels: np.ndarray) -> List[str]:
    """Return colors for user-provided labels using infoteam palette.
    
    Treats NaN entries as unlabeled (gray). Uses discrete color palette for digits 0-9.
    """
    colors = []
    for label in labels:
        if np.isnan(label):
            colors.append(INFOTEAM_PALETTE[10])  # Gray for unlabeled
        else:
            idx = int(label) % 10
            colors.append(INFOTEAM_PALETTE[idx])
    return colors


def _colorize_discrete_classes(classes: np.ndarray) -> List[str]:
    """Return colors for discrete class predictions using infoteam palette."""
    return [INFOTEAM_PALETTE[int(c) % 10] for c in classes]


def _colorize_numeric(values: np.ndarray) -> List[str]:
    """Return colors for an array of numeric values using viridis (colorblind-safe)."""
    return _values_to_hex_viridis(values)


def array_to_base64(arr: np.ndarray) -> str:
    """Convert a single-channel image array into a base64 encoded PNG data URI."""
    arr = np.array(arr)
    if arr.ndim == 1 and arr.size == 28 * 28:
        arr = arr.reshape(28, 28)
    if arr.ndim == 3:
        arr = arr.squeeze()

    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    if max_val > min_val:
        scaled = ((arr - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
    else:
        scaled = np.zeros_like(arr, dtype=np.uint8)

    img = Image.fromarray(scaled)
    if img.mode != "L":
        img = img.convert("L")

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _format_label_value(value: float | int | None, *, empty_text: str) -> str:
    """Return a compact string for hover metadata values."""
    if value is None:
        return empty_text
    if isinstance(value, float) and np.isnan(value):
        return empty_text
    return f"{int(value)}"


def _format_hover_metadata_entry(
    idx: int,
    pred_class: int,
    pred_certainty: float,
    user_label: float,
    true_label: int | None,
) -> List[object]:
    """Create the metadata payload for a single latent point."""
    return [
        int(idx),
        int(pred_class),
        float(pred_certainty) * 100.0,
        _format_label_value(user_label, empty_text="Unlabeled"),
        _format_label_value(true_label, empty_text="?"),
    ]


def _build_hover_metadata(
    pred_classes: np.ndarray,
    pred_certainty: np.ndarray,
    labels: np.ndarray,
    true_labels: np.ndarray | None,
) -> List[List[object]]:
    """Construct compact hover metadata for the latent scatter plot."""
    metadata: List[List[object]] = []
    total = int(len(pred_classes))
    for idx in range(total):
        true_label_val = None
        if true_labels is not None:
            true_label_val = int(true_labels[idx])
        metadata.append(
            _format_hover_metadata_entry(
                idx,
                int(pred_classes[idx]),
                float(pred_certainty[idx]),
                float(labels[idx]),
                true_label_val,
            )
        )
    return metadata


def compute_ema_smoothing(values: List[float], alpha: float = 0.15) -> List[float]:
    """Compute exponential moving average for smoothing loss curves.
    
    Args:
        values: Raw metric values
        alpha: Smoothing factor (0 = no smoothing, 1 = no memory)
    
    Returns:
        Smoothed values with same length as input
    """
    if not values or len(values) < 2:
        return list(values)
    
    smoothed = [values[0]]
    for val in values[1:]:
        smoothed.append(alpha * val + (1 - alpha) * smoothed[-1])
    return smoothed