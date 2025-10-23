"""Visualization callbacks for latent space interactions."""

from __future__ import annotations

from typing import Dict

from dash import Dash, Input, Output
from dash.exceptions import PreventUpdate
import numpy as np
import plotly.graph_objects as go

from use_cases.dashboard.state import app_state, state_lock
from use_cases.dashboard.utils import _colorize_numeric, _colorize_user_labels


def register_visualization_callbacks(app: Dash) -> None:
    """Register callbacks that control the latent space visualization."""

    @app.callback(
        Output("latent-scatter", "figure"),
        Input("color-mode-radio", "value"),
        Input("selected-sample-store", "data"),
        Input("labels-store", "data"),
        Input("latent-store", "data"),
    )
    def update_scatter(color_mode: str, selected_idx: int, _labels_store: dict, _latent_store: dict) -> go.Figure:
        with state_lock:
            latent = np.array(app_state["data"]["latent"], dtype=np.float64)
            labels = np.array(app_state["data"]["labels"], dtype=np.float64)
            true_labels = (
                np.array(app_state["data"]["true_labels"], dtype=np.float64)
                if app_state["data"]["true_labels"] is not None
                else None
            )
            pred_classes = np.array(app_state["data"]["pred_classes"], dtype=np.int32)
            pred_certainty = np.array(app_state["data"]["pred_certainty"], dtype=np.float64)
            hover_text = list(app_state["data"]["hover_text"])

        if latent is None or latent.size == 0:
            return go.Figure()

        figure = go.Figure()

        marker_kwargs: Dict[str, object]
        if color_mode == "user_labels":
            colors = _colorize_user_labels(labels)
            marker_kwargs = {"color": colors, "size": 7, "opacity": 0.9, "line": {"width": 0}}
        elif color_mode == "pred_class":
            colors = _colorize_numeric(pred_classes.astype(np.float64))
            marker_kwargs = {"color": colors, "size": 7, "opacity": 0.9, "line": {"width": 0}}
        elif color_mode == "true_class" and true_labels is not None:
            colors = _colorize_numeric(true_labels)
            marker_kwargs = {"color": colors, "size": 7, "opacity": 0.9, "line": {"width": 0}}
        elif color_mode == "certainty":
            colors = _colorize_numeric(pred_certainty)
            marker_kwargs = {"color": colors, "size": 7, "opacity": 0.95, "line": {"width": 0}}
        else:
            colors = _colorize_numeric(pred_classes.astype(np.float64))
            marker_kwargs = {"color": colors, "size": 7, "opacity": 0.9, "line": {"width": 0}}

        x_vals = latent[:, 0].astype(float).tolist()
        y_vals = latent[:, 1].astype(float).tolist()

        figure.add_trace(
            go.Scattergl(
                x=x_vals,
                y=y_vals,
                mode="markers",
                marker=marker_kwargs,
                hovertext=hover_text,
                hoverinfo="text",
            )
        )

        if selected_idx is not None and 0 <= int(selected_idx) < latent.shape[0]:
            point = (float(latent[int(selected_idx), 0]), float(latent[int(selected_idx), 1]))
            figure.add_trace(
                go.Scattergl(
                    x=[point[0]],
                    y=[point[1]],
                    mode="markers",
                    marker={
                        "color": "#ff0000",
                        "size": 14,
                        "symbol": "x",
                        "line": {"width": 2, "color": "#ffffff"},
                    },
                    name="Selected",
                    hoverinfo="skip",
                )
            )

        figure.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Latent Dimension 1",
            yaxis_title="Latent Dimension 2",
        )
        return figure

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

