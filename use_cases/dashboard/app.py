"""
Dash-based dashboard for the SSVAE active learning workflow (Phase 1).

This module implements the static UI with labeling capabilities:
- 60k-point latent space visualization backed by Plotly Scattergl
- Sample selection with original/reconstructed image preview
- Immediate label persistence to ``data/mnist/labels.csv``

Phase 2/3 features (training integration, live metrics) will extend the
structures defined here without restructuring the module.
"""

from __future__ import annotations

import base64
import io
import sys
import threading
from pathlib import Path
from queue import Queue
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import dash
from dash import ALL, Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from matplotlib import cm, colors as mcolors

# Ensure repository imports work when running without installation.
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ssvae import SSVAE, SSVAEConfig  # noqa: E402
from training.interactive_trainer import InteractiveTrainer  # noqa: E402
from data.mnist import load_train_images_for_ssvae, load_mnist_splits  # noqa: E402


CHECKPOINT_PATH = ROOT_DIR / "artifacts" / "checkpoints" / "ssvae.ckpt"
LABELS_PATH = ROOT_DIR / "data" / "mnist" / "labels.csv"


state_lock = threading.Lock()
metrics_queue: Queue[Dict[str, float]] = Queue()

app_state: Dict[str, object] = {
    "model": None,
    "trainer": None,
    "config": None,
    "data": {
        "x_train": None,
        "labels": None,
        "true_labels": None,
        "latent": None,
        "reconstructed": None,
        "pred_classes": None,
        "pred_certainty": None,
        "hover_text": None,
    },
    "training": {
        "active": False,
        "thread": None,
        "target_epochs": 0,
    },
    "ui": {
        "selected_sample": 0,
        "color_mode": "user_labels",
        "labels_version": 0,
    },
    "history": {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
    },
}


COOLWARM_CMAP = cm.get_cmap("coolwarm")


def initialize_model_and_data() -> None:
    """Load model, dataset, labels, and derived predictions into memory."""
    with state_lock:
        if app_state["model"] is not None:
            return

    config = SSVAEConfig()
    model = SSVAE(input_dim=(28, 28), config=config)

    if CHECKPOINT_PATH.exists():
        model.load_model_weights(str(CHECKPOINT_PATH))
        model.weights_path = str(CHECKPOINT_PATH)

    trainer = InteractiveTrainer(model)

    x_train = load_train_images_for_ssvae(dtype=np.float32)
    (_, true_labels), _ = load_mnist_splits(normalize=True, reshape=False, dtype=np.float32)
    true_labels = np.asarray(true_labels, dtype=np.int32)

    latent, recon, pred_classes, pred_certainty = model.predict(x_train)

    labels_array = np.full(shape=(x_train.shape[0],), fill_value=np.nan, dtype=float)
    if LABELS_PATH.exists():
        labels_df = pd.read_csv(LABELS_PATH)
        if not labels_df.empty:
            serials = labels_df["Serial"].astype(int)
            label_values = pd.to_numeric(labels_df["label"], errors="coerce")
            valid_mask = serials.between(0, x_train.shape[0] - 1) & label_values.notna()
            labels_array[serials[valid_mask]] = label_values[valid_mask].to_numpy(dtype=float)

    hover_text = _build_hover_text(pred_classes, pred_certainty, labels_array, true_labels)

    with state_lock:
        app_state["model"] = model
        app_state["trainer"] = trainer
        app_state["config"] = config
        app_state["data"]["x_train"] = x_train
        app_state["data"]["labels"] = labels_array
        app_state["data"]["true_labels"] = true_labels
        app_state["data"]["latent"] = latent
        app_state["data"]["reconstructed"] = recon
        app_state["data"]["pred_classes"] = pred_classes
        app_state["data"]["pred_certainty"] = pred_certainty
        app_state["data"]["hover_text"] = hover_text
        app_state["ui"]["selected_sample"] = int(app_state["ui"]["selected_sample"])
        app_state["training"]["active"] = False
        app_state["training"]["thread"] = None
        app_state["training"]["target_epochs"] = 0


def _format_hover_entry(
    idx: int,
    pred_class: int,
    pred_certainty: float,
    user_label: float,
    true_label: int | None,
) -> str:
    label_text = "Unlabeled" if np.isnan(user_label) else f"{int(user_label)}"
    true_label_text = "?" if true_label is None else f"{true_label}"
    return (
        f"Index: {idx}<br>Prediction: {pred_class}"
        f"<br>Confidence: {pred_certainty * 100:.1f}%"
        f"<br>User Label: {label_text}"
        f"<br>True Label: {true_label_text}"
    )


def _build_hover_text(
    pred_classes: np.ndarray,
    pred_certainty: np.ndarray,
    labels: np.ndarray,
    true_labels: np.ndarray,
) -> list[str]:
    hover = []
    for idx in range(len(pred_classes)):
        true_label = int(true_labels[idx]) if true_labels is not None else None
        hover.append(
            _format_hover_entry(
                idx,
                int(pred_classes[idx]),
                float(pred_certainty[idx]),
                float(labels[idx]),
                true_label,
            )
        )
    return hover


def array_to_base64(arr: np.ndarray) -> str:
    """Convert a single-channel image array into a base64 PNG string."""
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
    img = Image.fromarray(scaled, mode="L")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _values_to_hex(values: np.ndarray) -> list[str]:
    """Map numeric values to hex colors using the coolwarm colormap."""
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


def _colorize_user_labels(labels: np.ndarray) -> list[str]:
    filled = np.where(np.isnan(labels), 4.5, labels)
    return _values_to_hex(filled)


def _colorize_numeric(values: np.ndarray) -> list[str]:
    return _values_to_hex(values)


def _load_labels_dataframe() -> pd.DataFrame:
    """Load the labels CSV (index=Serial) or create an empty frame."""
    if LABELS_PATH.exists():
        df = pd.read_csv(LABELS_PATH, index_col="Serial")
        if not df.empty:
            df.index = df.index.astype(int)
        else:
            df.index = df.index.astype(int, copy=False)
    else:
        df = pd.DataFrame(columns=["label"])
        df.index.name = "Serial"
    return df


def _persist_labels_dataframe(df: pd.DataFrame) -> None:
    df.index.name = "Serial"
    df.to_csv(LABELS_PATH)


def _update_label(sample_idx: int, new_label: float | None) -> Tuple[dict, str]:
    """Update label state and CSV, returning store payload and status message."""
    with state_lock:
        labels_array: np.ndarray = app_state["data"]["labels"]
        if new_label is None:
            labels_array[sample_idx] = np.nan
        else:
            labels_array[sample_idx] = float(new_label)

        df = _load_labels_dataframe()
        if new_label is None:
            if sample_idx in df.index:
                df = df.drop(sample_idx)
        else:
            df.loc[sample_idx, "label"] = int(new_label)
        _persist_labels_dataframe(df)

        app_state["data"]["labels"] = labels_array
        pred_classes = np.array(app_state["data"]["pred_classes"], dtype=np.int32)
        pred_certainty = np.array(app_state["data"]["pred_certainty"], dtype=np.float64)
        true_labels = app_state["data"]["true_labels"]
        hover_text = list(app_state["data"]["hover_text"])
        true_label_value = int(true_labels[sample_idx]) if true_labels is not None else None
        hover_text[sample_idx] = _format_hover_entry(
            sample_idx,
            int(pred_classes[sample_idx]),
            float(pred_certainty[sample_idx]),
            labels_array[sample_idx],
            true_label_value,
        )
        app_state["data"]["hover_text"] = hover_text
        app_state["ui"]["labels_version"] = int(app_state["ui"]["labels_version"]) + 1
        version_payload = {"version": app_state["ui"]["labels_version"]}

    if new_label is None:
        message = f"Removed label for sample {sample_idx}"
    else:
        message = f"Labeled sample {sample_idx} as {int(new_label)}"
    return version_payload, message


def create_app() -> Dash:
    initialize_model_and_data()

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=False,
    )
    app.title = "SSVAE Active Learning Dashboard"

    app.layout = dbc.Container(
        [
            dcc.Store(id="selected-sample-store", data=int(app_state["ui"]["selected_sample"])),
            dcc.Store(id="labels-store", data={"version": int(app_state["ui"]["labels_version"])}),
            html.H1("SSVAE Active Learning Dashboard", className="mb-4"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Color By"),
                                    dbc.CardBody(
                                        dbc.RadioItems(
                                            id="color-mode-radio",
                                            options=[
                                                {"label": "User Labels", "value": "user_labels"},
                                                {"label": "Predicted Class", "value": "pred_class"},
                                                {"label": "True Label", "value": "true_class"},
                                                {"label": "Certainty", "value": "certainty"},
                                            ],
                                            value="user_labels",
                                            inline=True,
                                        )
                                    ),
                                ],
                                className="mb-3",
                            ),
                            dcc.Graph(
                                id="latent-scatter",
                                style={"height": "650px"},
                                config={"displayModeBar": True, "scrollZoom": True, "responsive": True},
                            ),
                        ],
                        width=8,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H4(id="selected-sample-header", className="mb-3"),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.H6("Original"),
                                                    html.Img(
                                                        id="original-image",
                                                        style={"width": "100%", "maxWidth": "250px"},
                                                    ),
                                                ],
                                                width=6,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.H6("Reconstruction"),
                                                    html.Img(
                                                        id="reconstructed-image",
                                                        style={"width": "100%", "maxWidth": "250px"},
                                                    ),
                                                ],
                                                width=6,
                                            ),
                                        ],
                                        className="mb-3",
                                    ),
                                    html.Div(id="prediction-info", className="mb-3 fw-semibold"),
                                    html.Div(
                                        [
                                            dbc.Label("Assign Label"),
                                            dbc.ButtonGroup(
                                                [
                                                    dbc.Button(
                                                        str(digit),
                                                        id={"type": "label-button", "label": digit},
                                                        color="primary",
                                                        outline=True,
                                                        n_clicks=0,
                                                        size="sm",
                                                    )
                                                    for digit in range(10)
                                                ],
                                                className="flex-wrap gap-1",
                                            ),
                                            dbc.Button(
                                                "Delete Label",
                                                id="delete-label-button",
                                                color="danger",
                                                outline=True,
                                                n_clicks=0,
                                                size="sm",
                                                className="ms-2",
                                            ),
                                        ],
                                        className="mb-2",
                                    ),
                                    html.Div(id="label-feedback", className="text-muted"),
                                ]
                            ),
                        ],
                        width=4,
                    ),
                ]
            ),
        ],
        fluid=True,
        className="pt-3 pb-5",
    )

    register_callbacks(app)
    return app


def register_callbacks(app: Dash) -> None:
    """Register Dash callbacks for the Phase 1 dashboard."""

    @app.callback(
        Output("latent-scatter", "figure"),
        Input("color-mode-radio", "value"),
        Input("selected-sample-store", "data"),
        Input("labels-store", "data"),
    )
    def update_scatter(color_mode: str, selected_idx: int, _labels_store: dict) -> go.Figure:
        with state_lock:
            latent = np.array(app_state["data"]["latent"], dtype=np.float64)
            labels = np.array(app_state["data"]["labels"], dtype=np.float64)
            true_labels = np.array(app_state["data"]["true_labels"], dtype=np.float64) if app_state["data"]["true_labels"] is not None else None
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
        Output("selected-sample-header", "children"),
        Output("original-image", "src"),
        Output("reconstructed-image", "src"),
        Output("prediction-info", "children"),
        Input("selected-sample-store", "data"),
        Input("labels-store", "data"),
    )
    def update_sample_display(selected_idx: int, _labels_store: dict) -> Tuple[str, str, str, str]:
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
        Output("color-mode-radio", "value"),
        Input("color-mode-radio", "value"),
    )
    def sync_color_mode(color_mode: str) -> str:
        with state_lock:
            app_state["ui"]["color_mode"] = color_mode
        return color_mode


app = create_app()


if __name__ == "__main__":
    initialize_model_and_data()
    app.run_server(debug=False, host="0.0.0.0", port=8050)
