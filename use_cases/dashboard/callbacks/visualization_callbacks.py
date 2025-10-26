"""Visualization callbacks for latent space interactions."""

from __future__ import annotations

from typing import Dict, Tuple

from dash import Dash, Input, Output, Patch, html
from dash.exceptions import PreventUpdate
import numpy as np
import plotly.graph_objects as go

from use_cases.dashboard.state import app_state, state_lock
from use_cases.dashboard.utils import (
    _colorize_numeric,
    _colorize_user_labels,
    _colorize_discrete_classes,
    compute_ema_smoothing,
    INFOTEAM_PALETTE,
)

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
        return _colorize_discrete_classes(pred_classes)
    if color_mode == "true_class" and true_labels is not None:
        return _colorize_discrete_classes(true_labels.astype(np.int32))
    if color_mode == "certainty":
        return _colorize_numeric(pred_certainty)
    return _colorize_discrete_classes(pred_classes)


def _build_marker(color_mode: str, colors: list[str]) -> Dict[str, object]:
    # Balance visibility vs overplotting (60k points)
    if color_mode == "certainty":
        opacity = 0.6  # Lower for continuous gradient
        size = 6
    else:
        opacity = 0.75  # Higher for discrete classes (better visibility)
        size = 7  # Slightly larger for discrete modes
    return {"color": colors, "size": size, "opacity": opacity, "line": {"width": 0}}


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
            showlegend=False,
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
            showlegend=False,
        )
    )

    figure.update_layout(
        template="plotly_white",
        margin=dict(l=50, r=20, t=20, b=50),
        xaxis_title=dict(text="Latent Dimension 1", font=dict(size=15, color="#000000", family="'Open Sans', Verdana, sans-serif")),
        yaxis_title=dict(text="Latent Dimension 2", font=dict(size=15, color="#000000", family="'Open Sans', Verdana, sans-serif")),
        dragmode="pan",
        hovermode="closest",
        uirevision=f"latent-{latent_version}",
        transition={"duration": 0},
        font=dict(size=13),  # Increase tick label font size
    )
    
    # Add visible grid lines
    figure.update_xaxes(
        showgrid=True,
        gridcolor="rgba(0, 0, 0, 0.1)",
        gridwidth=1,
        tickfont=dict(size=13),
    )
    figure.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0, 0, 0, 0.1)",
        gridwidth=1,
        tickfont=dict(size=13),
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

        # Clear old caches when latent changes
        if latent_version not in _BASE_FIGURE_CACHE:
            stale_keys = [key for key in _COLOR_CACHE if key[0] != latent_version]
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
        Output("scatter-legend", "children"),
        Input("color-mode-radio", "value"),
        Input("labels-store", "data"),
    )
    def update_legend(color_mode: str, _labels_store: dict | None):
        """Generate dynamic legend based on color mode."""
        if color_mode == "certainty":
            # Continuous colorbar - show range with infoteam-inspired gradient
            return html.Div(
                [
                    html.Span("0% (uncertain)", style={
                        "fontSize": "14px",
                        "color": "#6F6F6F",
                        "marginRight": "8px",
                        "fontFamily": "'Open Sans', Verdana, sans-serif",
                    }),
                    html.Div(style={
                        "flex": "1",
                        "height": "10px",
                        "background": "linear-gradient(to right, #6F6F6F, #8BBFC2, #AFCC37, #C10A27)",
                        "borderRadius": "4px",
                        "maxWidth": "200px",
                    }),
                    html.Span("100% (confident)", style={
                        "fontSize": "14px",
                        "color": "#6F6F6F",
                        "marginLeft": "8px",
                        "fontFamily": "'Open Sans', Verdana, sans-serif",
                    }),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "8px",
                    "padding": "8px 0",
                }
            )
        
        # Discrete legend for classes
        legend_items = []
        for i in range(10):
            legend_items.append(
                html.Div(
                    [
                        html.Div(style={
                            "width": "14px",
                            "height": "14px",
                            "backgroundColor": INFOTEAM_PALETTE[i],
                            "borderRadius": "3px",
                            "marginRight": "6px",
                        }),
                        html.Span(str(i), style={
                            "fontSize": "15px",
                            "color": "#4A4A4A",
                            "fontFamily": "ui-monospace, monospace",
                            "fontWeight": "500",
                        }),
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "center",
                    }
                )
            )
        
        # Add "unlabeled" for user_labels mode
        if color_mode == "user_labels":
            legend_items.append(
                html.Div(
                    [
                        html.Div(style={
                            "width": "14px",
                            "height": "14px",
                            "backgroundColor": INFOTEAM_PALETTE[10],  # Gray
                            "borderRadius": "3px",
                            "marginRight": "6px",
                        }),
                        html.Span("unlabeled", style={
                            "fontSize": "14px",
                            "color": "#6F6F6F",
                            "fontFamily": "ui-monospace, monospace",
                        }),
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "center",
                    }
                )
            )
        
        return html.Div(
            legend_items,
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "gap": "14px",
                "padding": "8px 0",
            }
        )

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
        Input("loss-smoothing-toggle", "value"),
    )
    def update_loss_curves(_latent_store: dict | None, smoothing_enabled: list):
        with state_lock:
            history = app_state["history"]
            epochs = list(history.get("epochs", []))

        figure = go.Figure()
        if not epochs:
            figure.update_layout(
                template="plotly_white",
                xaxis_title=dict(text="Epoch", font=dict(size=14)),
                yaxis_title=dict(text="Loss", font=dict(size=14)),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=12)),
                margin=dict(l=40, r=20, t=30, b=40),
                font=dict(size=12),
            )
            figure.update_xaxes(tickfont=dict(size=12))
            figure.update_yaxes(tickfont=dict(size=12))
            return figure

        # Check if smoothing is enabled (checkbox returns list of checked values)
        apply_smoothing = bool(smoothing_enabled)

        series_info = [
            ("train_loss", "Train Loss", "#C10A27"),
            ("val_loss", "Val Loss", "#45717A"),
            ("train_reconstruction_loss", "Train Recon", "#AFCC37"),
            ("val_reconstruction_loss", "Val Recon", "#F6C900"),
        ]

        for key, label, color in series_info:
            values = history.get(key)
            if values and len(values) == len(epochs):
                raw_values = list(values)
                
                # Add raw trace (solid line)
                figure.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=raw_values,
                        mode="lines+markers",
                        name=label,
                        line=dict(color=color, width=2),
                        marker=dict(size=4),
                    )
                )
                
                # Add smoothed trace if enabled and we have enough points
                if apply_smoothing and len(raw_values) >= 2:
                    smoothed = compute_ema_smoothing(raw_values, alpha=0.15)
                    figure.add_trace(
                        go.Scatter(
                            x=epochs,
                            y=smoothed,
                            mode="lines",
                            name=f"{label} (smoothed)",
                            line=dict(color=color, width=2, dash="dash"),
                            showlegend=True,
                        )
                    )

        figure.update_layout(
            template="plotly_white",
            xaxis_title=dict(text="Epoch", font=dict(size=14)),
            yaxis_title=dict(text="Loss", font=dict(size=14)),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                font=dict(size=12),
            ),
            margin=dict(l=40, r=20, t=40, b=40),
            font=dict(size=12),  # Increase tick label font size
        )
        figure.update_xaxes(tickfont=dict(size=12))
        figure.update_yaxes(tickfont=dict(size=12))
        return figure