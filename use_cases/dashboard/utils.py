"""Utility helpers for dashboard visualizations and rendering."""

from __future__ import annotations

import base64
import io
from typing import List

import numpy as np
from matplotlib import colormaps, colors as mcolors
from PIL import Image

COOLWARM_CMAP = colormaps["coolwarm"]


def _values_to_hex(values: np.ndarray) -> List[str]:
    """Map numeric values onto hex color codes using the coolwarm colormap."""
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
    rgba = COOLWARM_CMAP(normed)
    return [mcolors.to_hex(color[:3]) for color in rgba]


def _colorize_user_labels(labels: np.ndarray) -> List[str]:
    """Return colors for user-provided labels, treating NaN entries as unlabeled."""
    filled = np.where(np.isnan(labels), 4.5, labels)
    return _values_to_hex(filled)


def _colorize_numeric(values: np.ndarray) -> List[str]:
    """Return colors for an array of numeric values."""
    return _values_to_hex(values)


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


def _build_hover_text(
    pred_classes: np.ndarray,
    pred_certainty: np.ndarray,
    labels: np.ndarray,
    true_labels: np.ndarray,
) -> List[str]:
    """Construct hover tooltips for latent space scatter plot points."""
    hover_entries: List[str] = []
    for idx in range(len(pred_classes)):
        pred_class = int(pred_classes[idx])
        certainty = float(pred_certainty[idx])
        user_label = float(labels[idx])
        true_label = None
        if true_labels is not None:
            true_label = int(true_labels[idx])

        label_text = "Unlabeled" if np.isnan(user_label) else f"{int(user_label)}"
        true_label_text = "?" if true_label is None else f"{true_label}"
        hover_entries.append(
            f"Index: {idx}<br>Prediction: {pred_class}"
            f"<br>Confidence: {certainty * 100:.1f}%"
            f"<br>User Label: {label_text}"
            f"<br>True Label: {true_label_text}"
        )
    return hover_entries

