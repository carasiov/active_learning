"""Labeling callbacks for sample inspection and annotation."""

from __future__ import annotations

from typing import Tuple

import dash
from dash import ALL, Dash, Input, Output, State, html
from dash.exceptions import PreventUpdate
import numpy as np

from use_cases.dashboard.state import app_state, state_lock, _update_label
from use_cases.dashboard.utils import array_to_base64


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
    def update_sample_display(selected_idx: int, _labels_store: dict, _latent_store: dict) -> Tuple[str, str, str, str]:
        if selected_idx is None:
            raise PreventUpdate

        with state_lock:
            idx = int(selected_idx)
            x_train = np.array(app_state["data"]["x_train"])
            recon = np.array(app_state["data"]["reconstructed"])
            pred_classes = np.array(app_state["data"]["pred_classes"])
            pred_certainty = np.array(app_state["data"]["pred_certainty"])
            labels = np.array(app_state["data"]["labels"])
            true_labels = (
                np.array(app_state["data"]["true_labels"], dtype=np.int32)
                if app_state["data"]["true_labels"] is not None
                else None
            )

        if idx < 0 or idx >= x_train.shape[0]:
            raise PreventUpdate

        original_src = array_to_base64(x_train[idx])
        reconstructed_src = array_to_base64(recon[idx])

        user_label = labels[idx]
        label_text = "Unlabeled" if np.isnan(user_label) else f"{int(user_label)}"
        true_label = int(true_labels[idx]) if true_labels is not None else "?"
        prediction_text = (
            f"Predicted: {int(pred_classes[idx])} "
            f"({pred_certainty[idx] * 100:.1f}% confidence) | "
            f"User Label: {label_text} | True Label: {true_label}"
        )
        header_text = f"Selected Sample #{idx}"
        return header_text, original_src, reconstructed_src, prediction_text

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
        if triggered_id == "delete-label-button":
            version_payload, message = _update_label(int(selected_idx), None)
            return version_payload, message
        if isinstance(triggered_id, dict) and "label" in triggered_id:
            label_value = int(triggered_id["label"])
            version_payload, message = _update_label(int(selected_idx), label_value)
            return version_payload, message
        raise PreventUpdate

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
        version_payload, message = _update_label(int(selected_idx), int(digit))
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
        with state_lock:
            labels = np.array(app_state["data"]["labels"], dtype=float)

        total_samples = int(labels.size)
        labeled_mask = ~np.isnan(labels)
        labeled_count = int(np.sum(labeled_mask))
        unlabeled_count = total_samples - labeled_count
        labeled_pct = (labeled_count / total_samples * 100.0) if total_samples else 0.0

        lines = [
            html.Div(f"Total samples: {total_samples}"),
            html.Div(f"Labeled: {labeled_count} ({labeled_pct:.1f}%)"),
            html.Div(f"Unlabeled: {unlabeled_count}"),
        ]

        if labeled_count > 0:
            labeled_values = labels[labeled_mask].astype(int)
            lines.append(html.Hr(className="my-2"))
            lines.append(html.Div("Label distribution:", className="fw-bold"))
            for digit in range(10):
                count = int(np.sum(labeled_values == digit))
                class_name = None if count > 0 else "text-muted"
                lines.append(html.Div(f"{digit}: {count}", className=class_name))

        return lines
