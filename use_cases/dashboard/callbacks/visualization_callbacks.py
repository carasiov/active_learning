"""Visualization callbacks for latent space interactions."""

from __future__ import annotations

from typing import Dict, Tuple

from dash import Dash, Input, Output, Patch
from dash.exceptions import PreventUpdate
import numpy as np
import plotly.graph_objects as go

from use_cases.dashboard.state import app_state, state_lock
from use_cases.dashboard.utils import _colorize_numeric, _colorize_user_labels

_BASE_FIGURE_CACHE: Dict[int, go.Figure] = {}
_COLOR_CACHE: Dict[Tuple[int, str, int], list[str]] = {}

_LATENT_HOVER_TEMPLATE = (
    "Index: %{customdata[0]}<br>"
    "Prediction: %{customdata[1]}<br>"
    "Confidence: %{customdata[2]:.1f}%<br>"
    "User Label: %{customdata[3]}<br>"
    "True Label: %{customdata[4]}<extra></extra>"
)


def _compute_colors(
    color_mode: str,
    labels: np.ndarray,
    pred_classes: np.ndarray,
    pred_certainty: np.ndarray,
    true_labels: np.ndarray | None,
) -> list[str]:
    if color_mode == "user_labels":
        return _colorize_user_labels(labels)
    if color_mode == "pred_class":
        return _colorize_numeric(pred_classes.astype(np.float64))
    if color_mode == "true_class" and true_labels is not None:
        return _colorize_numeric(true_labels)
    if color_mode == "certainty":
        return _colorize_numeric(pred_certainty)
    return _colorize_numeric(pred_classes.astype(np.float64))


def _build_marker(color_mode: str, colors: list[str]) -> Dict[str, object]:
    opacity = 0.95 if color_mode == "certainty" else 0.9
    return {"color": colors, "size": 7, "opacity": opacity, "line": {"width": 0}}


def _build_highlight(latent: np.ndarray, selected_idx: int | None) -> Tuple[list[float], list[float], bool]:
    if selected_idx is None or selected_idx < 0 or selected_idx >= latent.shape[0]:
        return [], [], False
    point = latent[int(selected_idx)]
    return [float(point[0])], [float(point[1])], True


def _build_base_figure(
    latent: np.ndarray,
    hover_metadata: list[list[object]],
    color_mode: str,
    colors: list[str],
    selected_idx: int | None,
    latent_version: int,
) -> go.Figure:
    x_vals = latent[:, 0]
    y_vals = latent[:, 1]

    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker=_build_marker(color_mode, colors),
            customdata=hover_metadata,
            hovertemplate=_LATENT_HOVER_TEMPLATE,
        )
    )

    highlight_x, highlight_y, visible = _build_highlight(latent, selected_idx)
    figure.add_trace(
        go.Scattergl(
            x=highlight_x,
            y=highlight_y,
            mode="markers",
            marker={
                "color": "#ff0000",
                "size": 14,
                "symbol": "x",
                "line": {"width": 2, "color": "#ffffff"},
            },
            name="Selected",
            hoverinfo="skip",
            visible=visible,
        )
    )

    figure.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Latent Dimension 1",
        yaxis_title="Latent Dimension 2",
        dragmode="pan",
        hovermode="closest",
        uirevision=f"latent-{latent_version}",
        transition={"duration": 0},
    )
    return figure


def register_visualization_callbacks(app: Dash) -> None:
    """Register callbacks that control the latent space visualization."""

    @app.callback(
        Output("latent-scatter", "figure"),
        Input("color-mode-radio", "value"),
        Input("selected-sample-store", "data"),
        Input("labels-store", "data"),
        Input("latent-store", "data"),
    )
    def update_scatter(
        color_mode: str,
        selected_idx: int | None,
        labels_store: dict | None,
        latent_store: dict | None,
    ):
        latent_version = int((latent_store or {}).get("version", 0))
        label_version = int((labels_store or {}).get("version", 0))

        with state_lock:
            latent = np.array(app_state["data"]["latent"], dtype=np.float32)
            labels = np.array(app_state["data"]["labels"], dtype=np.float64)
            true_labels = (
                np.array(app_state["data"]["true_labels"], dtype=np.float64)
                if app_state["data"]["true_labels"] is not None
                else None
            )
            pred_classes = np.array(app_state["data"]["pred_classes"], dtype=np.int32)
            pred_certainty = np.array(app_state["data"]["pred_certainty"], dtype=np.float64)
            hover_metadata = list(app_state["data"].get("hover_metadata", []))

        if latent is None or latent.size == 0:
            return go.Figure()
        if not hover_metadata:
            hover_metadata = [
                [idx, 0, 0.0, "Unlabeled", "?"] for idx in range(latent.shape[0])
            ]

        # Drop stale caches for old latent versions to keep memory bounded.
        if latent_version not in _BASE_FIGURE_CACHE:
            stale_keys = [key for key in _COLOR_CACHE if key[0] == latent_version]
            for key in stale_keys:
                _COLOR_CACHE.pop(key, None)

        cache_key = (latent_version, color_mode, label_version)
        colors = _COLOR_CACHE.get(cache_key)
        if colors is None:
            colors = _compute_colors(color_mode, labels, pred_classes, pred_certainty, true_labels)
            _COLOR_CACHE[cache_key] = colors

        if latent_version not in _BASE_FIGURE_CACHE:
            figure = _build_base_figure(
                latent,
                hover_metadata,
                color_mode,
                colors,
                selected_idx,
                latent_version,
            )
            _BASE_FIGURE_CACHE[latent_version] = figure
            return figure

        highlight_x, highlight_y, visible = _build_highlight(latent, selected_idx)
        patch = Patch()
        patch["data"][0]["marker"] = _build_marker(color_mode, colors)
        patch["data"][0]["customdata"] = hover_metadata
        patch["data"][1]["x"] = highlight_x
        patch["data"][1]["y"] = highlight_y
        patch["data"][1]["visible"] = visible
        patch["data"][1]["marker"] = {
            "color": "#ff0000",
            "size": 14,
            "symbol": "x",
            "line": {"width": 2, "color": "#ffffff"},
        }
        patch["layout"] = {"uirevision": f"latent-{latent_version}"}
        return patch

    @app.callback(
        Output("selected-sample-store", "data"),
        Input("latent-scatter", "clickData"),
        prevent_initial_call=True,
    )
    def handle_point_selection(click_data: Dict) -> int:
        if not click_data or "points" not in click_data or not click_data["points"]:
            raise PreventUpdate
        point_index = int(click_data["points"][0]["pointIndex"])
        with state_lock:
            app_state["ui"]["selected_sample"] = point_index
        return point_index

    @app.callback(
        Output("color-mode-radio", "value"),
        Input("color-mode-radio", "value"),
    )
    def sync_color_mode(color_mode: str) -> str:
        with state_lock:
            app_state["ui"]["color_mode"] = color_mode
        return color_mode

    @app.callback(
        Output("loss-curves", "figure"),
        Input("latent-store", "data"),
    )
    def update_loss_curves(_latent_store: dict | None):
        with state_lock:
            history = app_state["history"]
            epochs = list(history.get("epochs", []))

        figure = go.Figure()
        if not epochs:
            figure.update_layout(
                template="plotly_white",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            return figure

        series_info = [
            ("train_loss", "Train Loss"),
            ("val_loss", "Val Loss"),
            ("train_reconstruction_loss", "Train Recon"),
            ("val_reconstruction_loss", "Val Recon"),
        ]

        for key, label in series_info:
            values = history.get(key)
            if values and len(values) == len(epochs):
                figure.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=list(values),
                        mode="lines+markers",
                        name=label,
                    )
                )

        figure.update_layout(
            template="plotly_white",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=20, r=20, t=30, b=40),
        )
        return figure
