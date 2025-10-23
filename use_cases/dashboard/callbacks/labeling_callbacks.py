"""Labeling callbacks for sample inspection and annotation."""

from __future__ import annotations

from typing import Tuple

import dash
from dash import ALL, Dash, Input, Output, State
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

