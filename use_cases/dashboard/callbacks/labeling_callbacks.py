"""Labeling callbacks with improved formatting."""

from __future__ import annotations

from typing import Tuple

import dash
from dash import ALL, Dash, Input, Output, State, dcc
from dash.exceptions import PreventUpdate
import numpy as np
import plotly.graph_objects as go

from use_cases.dashboard.core import state as dashboard_state
from use_cases.dashboard.core.commands import LabelSampleCommand
from use_cases.dashboard.utils.visualization import array_to_base64, INFOTEAM_PALETTE


def register_labeling_callbacks(app: Dash) -> None:
    """Register callbacks handling sample display and label updates."""

    @app.callback(
        Output("selected-sample-header", "children"),
        Output("original-image", "src"),
        Output("reconstructed-image", "src"),
        Output("prediction-info", "children"),
        Input("selected-sample-store", "data"),
        Input("labels-store", "data"),
        Input("latent-store", "data"),
    )
    def update_sample_display(selected_idx: int, _labels_store: dict, _latent_store: dict) -> Tuple[str, str, str, object]:
        if selected_idx is None:
            raise PreventUpdate

        with dashboard_state.state_lock:
            if dashboard_state.app_state.active_model is None:
                raise PreventUpdate
            idx = int(selected_idx)
            x_train = np.array(dashboard_state.app_state.active_model.data.x_train)
            recon = np.array(dashboard_state.app_state.active_model.data.reconstructed)
            pred_classes = np.array(dashboard_state.app_state.active_model.data.pred_classes)
            pred_certainty = np.array(dashboard_state.app_state.active_model.data.pred_certainty)
            labels = np.array(dashboard_state.app_state.active_model.data.labels)
            true_labels = (
                np.array(dashboard_state.app_state.active_model.data.true_labels, dtype=np.int32)
                if dashboard_state.app_state.active_model.data.true_labels is not None
                else None
            )

        if idx < 0 or idx >= x_train.shape[0]:
            raise PreventUpdate

        original_src = array_to_base64(x_train[idx])
        reconstructed_src = array_to_base64(recon[idx])

        user_label = labels[idx]
        label_text = "unlabeled" if np.isnan(user_label) else f"{int(user_label)}"
        true_label = int(true_labels[idx]) if true_labels is not None else "?"
        
        # Format as newlines for better readability
        prediction_info = dash.html.Div(
            [
                dash.html.Div(f"Predicted:  {int(pred_classes[idx])} ({pred_certainty[idx] * 100:.1f}%)", style={
                    "marginBottom": "4px",
                }),
                dash.html.Div(f"User Label: {label_text}", style={
                    "marginBottom": "4px",
                }),
                dash.html.Div(f"True Label: {true_label}"),
            ],
            style={
                "fontFamily": "ui-monospace, 'SF Mono', Monaco, monospace",
                "fontSize": "14px",
                "color": "#4A4A4A",
                "lineHeight": "1.6",
            }
        )
        
        header_text = f"Sample #{idx}"
        return header_text, original_src, reconstructed_src, prediction_info

    @app.callback(
        Output("labels-store", "data"),
        Output("label-feedback", "children"),
        Input({"type": "label-button", "label": ALL}, "n_clicks"),
        Input("delete-label-button", "n_clicks"),
        State("selected-sample-store", "data"),
        prevent_initial_call=True,
    )
    def handle_label_actions(label_clicks, delete_clicks: int, selected_idx: int) -> Tuple[dict, str]:
        if selected_idx is None:
            raise PreventUpdate
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        triggered_id = ctx.triggered_id
        
        # Determine command based on trigger
        if triggered_id == "delete-label-button":
            command = LabelSampleCommand(sample_idx=int(selected_idx), label=None)
        elif isinstance(triggered_id, dict) and "label" in triggered_id:
            label_value = int(triggered_id["label"])
            command = LabelSampleCommand(sample_idx=int(selected_idx), label=label_value)
        else:
            raise PreventUpdate
        
        # Execute command
        success, message = dashboard_state.dispatcher.execute(command)
        
        # Return updated version
        with dashboard_state.state_lock:
            if dashboard_state.app_state.active_model:
                version_payload = {"version": dashboard_state.app_state.active_model.data.version}
            else:
                version_payload = {"version": 0}
        
        return version_payload, message

    @app.callback(
        Output("labels-store", "data", allow_duplicate=True),
        Output("label-feedback", "children", allow_duplicate=True),
        Input("keyboard-label-store", "data"),
        State("selected-sample-store", "data"),
        prevent_initial_call=True,
    )
    def handle_keyboard_label(event: dict | None, selected_idx: int | None) -> Tuple[dict, str]:
        if not event or selected_idx is None:
            raise PreventUpdate
        digit = event.get("digit")
        if digit is None or not (0 <= int(digit) <= 9):
            raise PreventUpdate
        
        # Execute command
        command = LabelSampleCommand(sample_idx=int(selected_idx), label=int(digit))
        success, message = dashboard_state.dispatcher.execute(command)
        
        # Return updated version
        with dashboard_state.state_lock:
            if dashboard_state.app_state.active_model:
                version_payload = {"version": dashboard_state.app_state.active_model.data.version}
            else:
                version_payload = {"version": 0}
        
        return version_payload, message

    app.clientside_callback(
        """
        function(n, existing) {
            if (typeof window.__dashKeyboardListener__ === "undefined") {
                window.__dashKeyboardListener__ = {digit: null, timestamp: null};
                document.addEventListener("keydown", function(evt) {
                    if (evt.metaKey || evt.ctrlKey || evt.altKey) {
                        return;
                    }
                    const tag = (evt.target && evt.target.tagName) ? evt.target.tagName.toLowerCase() : "";
                    if (tag === "input" || tag === "textarea" || (evt.target && evt.target.isContentEditable)) {
                        return;
                    }
                    const value = parseInt(evt.key, 10);
                    if (!Number.isNaN(value) && value >= 0 && value <= 9) {
                        window.__dashKeyboardListener__ = {digit: value, timestamp: Date.now()};
                    }
                });
            }
            const last = window.__dashKeyboardListener__;
            if (!last || last.timestamp === null) {
                return window.dash_clientside.no_update;
            }
            if (existing && existing.timestamp === last.timestamp) {
                return window.dash_clientside.no_update;
            }
            return last;
        }
        """,
        Output("keyboard-label-store", "data"),
        Input("keyboard-poll", "n_intervals"),
        State("keyboard-label-store", "data"),
        prevent_initial_call=False,
    )

    @app.callback(
        Output("dataset-stats", "children"),
        Input("labels-store", "data"),
    )
    def update_dataset_stats(_labels_store: dict | None):
        with dashboard_state.state_lock:
            if dashboard_state.app_state.active_model:
                labels = np.array(dashboard_state.app_state.active_model.data.labels, dtype=float)
            else:
                labels = np.array([], dtype=float)

        total_samples = int(labels.size)
        labeled_mask = ~np.isnan(labels)
        labeled_count = int(np.sum(labeled_mask))
        unlabeled_count = total_samples - labeled_count
        labeled_pct = (labeled_count / total_samples * 100.0) if total_samples else 0.0

        # Highlight labeled percentage with larger/bold font
        stats_lines = [
            dash.html.Div([
                "Total: ",
                dash.html.Span(f"{total_samples:,}", style={"fontWeight": "600"}),
            ], style={
                "fontFamily": "ui-monospace, 'SF Mono', monospace",
                "fontSize": "14px",
                "color": "#4A4A4A",
                "lineHeight": "1.8",
            }),
            dash.html.Div([
                "Labeled: ",
                dash.html.Span(f"{labeled_count:,} / {total_samples:,}", style={"fontWeight": "600"}),
                dash.html.Span(f" ({labeled_pct:.1f}%)", style={
                    "fontWeight": "700",
                    "color": "#C10A27",
                    "fontSize": "15px",
                }),
            ], style={
                "fontFamily": "ui-monospace, 'SF Mono', monospace",
                "fontSize": "14px",
                "color": "#4A4A4A",
                "lineHeight": "1.8",
            }),
            dash.html.Div([
                "Unlabeled: ",
                dash.html.Span(f"{unlabeled_count:,}", style={"fontWeight": "600"}),
            ], style={
                "fontFamily": "ui-monospace, 'SF Mono', monospace",
                "fontSize": "14px",
                "color": "#4A4A4A",
                "lineHeight": "1.8",
                "marginBottom": "12px",
            }),
        ]

        # Label distribution histogram if we have labels
        if labeled_count > 0:
            stats_lines.append(dash.html.Hr(style={
                "margin": "16px 0",
                "border": "none",
                "borderTop": "1px solid #C6C6C6",
            }))
            
            stats_lines.append(dash.html.Div("Label Distribution", style={
                "fontSize": "14px",
                "fontWeight": "600",
                "color": "#6F6F6F",
                "marginBottom": "12px",
                "textTransform": "uppercase",
                "letterSpacing": "0.5px",
            }))
            
            # Count labels per digit
            labeled_values = labels[labeled_mask].astype(int)
            counts = [int(np.sum(labeled_values == digit)) for digit in range(10)]
            max_count = max(counts) if counts else 1
            
            # Create vertical bar chart with infoteam colors (rotated to horizontal display)
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=[str(i) for i in range(10)],
                y=counts,
                orientation='v',
                marker=dict(
                    color=[INFOTEAM_PALETTE[i] for i in range(10)],
                    line=dict(width=0),
                ),
                text=counts,
                textposition='outside',
                textfont=dict(size=11, color='#4A4A4A', family='ui-monospace, monospace'),
                hovertemplate='Digit %{x}: %{y} samples<extra></extra>',
            ))
            
            fig.update_layout(
                template="plotly_white",
                height=140,
                margin=dict(l=30, r=30, t=25, b=30),
                xaxis=dict(
                    title="",
                    tickfont=dict(size=12, color="#4A4A4A", family="ui-monospace, monospace"),
                    showgrid=False,
                ),
                yaxis=dict(
                    title="Count",
                    titlefont=dict(size=11, color="#6F6F6F", family="'Open Sans', Verdana, sans-serif"),
                    showgrid=True,
                    gridcolor="rgba(0, 0, 0, 0.05)",
                    tickfont=dict(size=10, color="#6F6F6F"),
                    range=[0, max(max_count * 1.2, 5)],  # Minimum range of 5 for visibility
                ),
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            
            stats_lines.append(
                dcc.Graph(
                    figure=fig,
                    config={'displayModeBar': False},
                    style={"marginTop": "8px"},
                )
            )

        return stats_lines