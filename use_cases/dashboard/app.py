"""
Dash-based dashboard for the SSVAE active learning workflow.

Currently includes:
- 60k-point latent space visualization backed by Plotly Scattergl
- Sample selection with original/reconstructed image preview
- Immediate label persistence to ``data/mnist/labels.csv``
- Background training controls with live status updates
"""

from __future__ import annotations

import base64
import io
import sys
import threading
from pathlib import Path
from queue import Queue, Empty
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import dash
from dash import ALL, Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from matplotlib import colormaps, colors as mcolors

# Ensure repository imports work when running without installation.
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
DASHBOARD_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from ssvae import SSVAE, SSVAEConfig  # noqa: E402
from training.interactive_trainer import InteractiveTrainer  # noqa: E402
from data.mnist import load_train_images_for_ssvae, load_mnist_splits  # noqa: E402
from dashboard_callback import DashboardMetricsCallback  # noqa: E402


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
        "status_messages": [],
    },
    "ui": {
        "selected_sample": 0,
        "color_mode": "user_labels",
        "labels_version": 0,
        "latent_version": 0,
    },
    "history": {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "train_reconstruction_loss": [],
        "val_reconstruction_loss": [],
        "train_kl_loss": [],
        "val_kl_loss": [],
        "train_classification_loss": [],
        "val_classification_loss": [],
    },
}


COOLWARM_CMAP = colormaps["coolwarm"]
MAX_STATUS_MESSAGES = 10


def _append_status_message_locked(message: str) -> None:
    messages = app_state["training"].setdefault("status_messages", [])
    messages.append(message)
    if len(messages) > MAX_STATUS_MESSAGES:
        messages = messages[-MAX_STATUS_MESSAGES:]
    app_state["training"]["status_messages"] = messages


def _append_status_message(message: str) -> None:
    with state_lock:
        _append_status_message_locked(message)


def _update_history_with_epoch(payload: Dict[str, float]) -> None:
    with state_lock:
        history = app_state["history"]
        history["epochs"].append(int(payload["epoch"]))
        for key in (
            "train_loss",
            "val_loss",
            "train_reconstruction_loss",
            "val_reconstruction_loss",
            "train_kl_loss",
            "val_kl_loss",
            "train_classification_loss",
            "val_classification_loss",
        ):
            value = payload.get(key)
            if value is not None:
                history.setdefault(key, []).append(float(value))


def _clear_metrics_queue() -> None:
    while True:
        try:
            metrics_queue.get_nowait()
        except Empty:
            break


def _configure_trainer_callbacks(trainer: InteractiveTrainer, target_epochs: int) -> None:
    existing = getattr(trainer, "_callbacks", None)
    if existing:
        base_callbacks = [cb for cb in existing if not isinstance(cb, DashboardMetricsCallback)]
    else:
        base_callbacks = list(
            trainer.model._build_callbacks(
                weights_path=trainer.model.weights_path or str(CHECKPOINT_PATH),
                export_history=False,
            )
        )
    base_callbacks.append(DashboardMetricsCallback(metrics_queue, target_epochs))
    trainer._callbacks = base_callbacks


def train_worker(num_epochs: int) -> None:
    """Background worker that runs incremental training and pushes updates to the UI."""
    try:
        with state_lock:
            trainer: InteractiveTrainer = app_state["trainer"]
            x_train_ref = app_state["data"]["x_train"]
            labels_ref = app_state["data"]["labels"]
            target_epochs = int(app_state["training"]["target_epochs"] or num_epochs)
            _configure_trainer_callbacks(trainer, target_epochs)

        if x_train_ref is None or labels_ref is None:
            metrics_queue.put({"type": "error", "message": "Training data not initialized."})
            return

        x_train = np.array(x_train_ref)
        labels = np.array(labels_ref, copy=True)

        _append_status_message(f"Training for {target_epochs} epoch(s)...")
        history = trainer.train_epochs(
            num_epochs=target_epochs,
            data=x_train,
            labels=labels,
            weights_path=str(CHECKPOINT_PATH),
        )

        with state_lock:
            model = app_state["model"]
        latent, recon, pred_classes, pred_certainty = model.predict(x_train)

        with state_lock:
            labels_latest = np.array(app_state["data"]["labels"], copy=True)
            true_labels = app_state["data"]["true_labels"]
        hover_text = _build_hover_text(pred_classes, pred_certainty, labels_latest, true_labels)

        with state_lock:
            app_state["data"]["latent"] = latent
            app_state["data"]["reconstructed"] = recon
            app_state["data"]["pred_classes"] = pred_classes
            app_state["data"]["pred_certainty"] = pred_certainty
            app_state["data"]["hover_text"] = hover_text
            app_state["ui"]["latent_version"] = int(app_state["ui"]["latent_version"]) + 1
            latent_version = app_state["ui"]["latent_version"]
        metrics_queue.put({"type": "latent_updated", "version": latent_version})
        metrics_queue.put({"type": "training_complete", "history": history})
    except Exception as exc:  # pragma: no cover - defensive
        metrics_queue.put({"type": "error", "message": str(exc)})
    finally:
        with state_lock:
            app_state["training"]["active"] = False
            app_state["training"]["thread"] = None


def initialize_model_and_data() -> None:
    """Load model, dataset, labels, and derived predictions into memory."""
    with state_lock:
        if app_state["model"] is not None:
            return
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

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
    stored_labels = _load_labels_dataframe()
    if not stored_labels.empty:
        serials = stored_labels.index.to_numpy()
        label_values = stored_labels["label"].astype(int).to_numpy()
        valid_mask = (serials >= 0) & (serials < x_train.shape[0])
        labels_array[serials[valid_mask]] = label_values[valid_mask].astype(float)

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
        app_state["training"]["status_messages"] = []
        app_state["ui"]["latent_version"] = 0


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
    img = Image.fromarray(scaled)
    if img.mode != "L":
        img = img.convert("L")
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
    """Load the labels CSV (index=Serial) or create an empty frame with Int64 labels."""
    columns = ["Serial", "label"]
    if LABELS_PATH.exists():
        df = pd.read_csv(LABELS_PATH, usecols=columns)
    else:
        df = pd.DataFrame(columns=columns)

    if df.empty:
        empty = pd.DataFrame(columns=["label"])
        empty.index = pd.Index([], name="Serial", dtype=int)
        empty["label"] = pd.Series(dtype="Int64")
        return empty

    df["Serial"] = pd.to_numeric(df["Serial"], errors="coerce")
    df = df.dropna(subset=["Serial"])
    df["Serial"] = df["Serial"].astype(int)
    df["label"] = pd.to_numeric(df.get("label"), errors="coerce").astype("Int64")
    df = df.set_index("Serial")
    df.index.name = "Serial"
    return df


def _persist_labels_dataframe(df: pd.DataFrame) -> None:
    persisted = df.copy()
    if not persisted.empty:
        persisted.index = persisted.index.astype(int)
        persisted["label"] = persisted["label"].astype("Int64")
    persisted.index.name = "Serial"
    persisted.to_csv(LABELS_PATH)


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

    with state_lock:
        config = app_state["config"]
        recon_weight_value = float(np.clip(config.recon_weight, 0.0, 5000.0))
        kl_weight_value = float(np.clip(config.kl_weight, 0.0, 1.0))
        learning_rate_value = float(np.clip(config.learning_rate, 0.0001, 0.01))
        default_epochs = max(1, int(app_state["training"]["target_epochs"] or 5))
        latent_version = int(app_state["ui"]["latent_version"])
        existing_status = list(app_state["training"]["status_messages"])

    status_initial_children = (
        html.Ul([html.Li(msg) for msg in existing_status], className="mb-0 small")
        if existing_status
        else html.Span("Idle.", className="text-muted")
    )

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
            dcc.Store(id="training-control-store", data={"token": 0}),
            dcc.Store(id="latent-store", data={"version": latent_version}),
            dcc.Interval(id="training-poll", interval=2000, n_intervals=0, disabled=True),
            html.H1("SSVAE Active Learning Dashboard", className="mb-4"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Training Controls"),
                                dbc.CardBody(
                                    [
                                        dbc.Label("Reconstruction Weight"),
                                        dcc.Slider(
                                            id="recon-weight-slider",
                                            min=0,
                                            max=5000,
                                            step=50,
                                            value=recon_weight_value,
                                            marks={0: "0", 2500: "2500", 5000: "5000"},
                                            tooltip={"placement": "bottom", "always_visible": False},
                                        ),
                                        dbc.Label("KL Weight", className="mt-3"),
                                        dcc.Slider(
                                            id="kl-weight-slider",
                                            min=0.0,
                                            max=1.0,
                                            step=0.01,
                                            value=kl_weight_value,
                                            marks={0.0: "0", 0.5: "0.5", 1.0: "1"},
                                            tooltip={"placement": "bottom", "always_visible": False},
                                        ),
                                        dbc.Label("Learning Rate", className="mt-3"),
                                        dcc.Slider(
                                            id="learning-rate-slider",
                                            min=0.0001,
                                            max=0.01,
                                            step=0.0001,
                                            value=learning_rate_value,
                                            marks={0.0001: "1e-4", 0.005: "5e-3", 0.01: "1e-2"},
                                            tooltip={"placement": "bottom", "always_visible": False},
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Epochs", className="mt-3"),
                                                        dcc.Input(
                                                            id="num-epochs-input",
                                                            type="number",
                                                            min=1,
                                                            max=200,
                                                            step=1,
                                                            value=default_epochs,
                                                            debounce=True,
                                                            style={"width": "100%"},
                                                        ),
                                                    ],
                                                    md=4,
                                                ),
                                                dbc.Col(
                                                    dbc.Button(
                                                        "Start Training",
                                                        id="start-training-button",
                                                        color="success",
                                                        className="mt-4",
                                                        n_clicks=0,
                                                    ),
                                                    md="auto",
                                                ),
                                            ],
                                            className="g-3 align-items-end",
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        width=8,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Training Status"),
                                dbc.CardBody(
                                    html.Div(status_initial_children, id="training-status", className="m-0"),
                                ),
                            ],
                            className="mb-3",
                        ),
                        width=4,
                    ),
                ],
                className="mb-2",
            ),
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
    """Register Dash callbacks for the dashboard."""

    @app.callback(
        Output("training-control-store", "data"),
        Input("start-training-button", "n_clicks"),
        State("recon-weight-slider", "value"),
        State("kl-weight-slider", "value"),
        State("learning-rate-slider", "value"),
        State("num-epochs-input", "value"),
        State("training-control-store", "data"),
        prevent_initial_call=True,
    )
    def handle_start_training(
        n_clicks: int,
        recon_weight: float,
        kl_weight: float,
        learning_rate: float,
        num_epochs: Optional[float],
        control_store: Optional[Dict[str, int]],
    ) -> object:
        if not n_clicks:
            raise PreventUpdate

        if num_epochs is None:
            _append_status_message("Please specify the number of epochs before starting training.")
            return dash.no_update

        try:
            epochs = int(num_epochs)
        except (TypeError, ValueError):
            _append_status_message("Epochs must be a whole number between 1 and 200.")
            return dash.no_update

        if epochs <= 0:
            _append_status_message("Epochs must be greater than zero.")
            return dash.no_update
        epochs = max(1, min(epochs, 200))

        if recon_weight is None or kl_weight is None or learning_rate is None:
            _append_status_message("Please configure all hyperparameters before training.")
            return dash.no_update

        with state_lock:
            if app_state["training"]["active"]:
                _append_status_message_locked("Training already in progress.")
                return dash.no_update

            config = app_state["config"]
            config.recon_weight = float(recon_weight)
            config.kl_weight = float(kl_weight)
            config.learning_rate = float(learning_rate)

            model = app_state["model"]
            model.config.recon_weight = float(recon_weight)
            model.config.kl_weight = float(kl_weight)
            model.config.learning_rate = float(learning_rate)

            trainer = app_state["trainer"]
            trainer.config.recon_weight = float(recon_weight)
            trainer.config.kl_weight = float(kl_weight)
            trainer.config.learning_rate = float(learning_rate)

            app_state["training"]["target_epochs"] = epochs
            app_state["training"]["active"] = True
            app_state["training"]["status_messages"] = [f"Queued training for {epochs} epoch(s)."]

        _clear_metrics_queue()
        worker = threading.Thread(target=train_worker, args=(epochs,), daemon=True)
        with state_lock:
            app_state["training"]["thread"] = worker
        worker.start()

        token = (control_store or {}).get("token", 0) + 1
        return {"token": token}

    @app.callback(
        Output("training-status", "children"),
        Output("start-training-button", "disabled"),
        Output("recon-weight-slider", "disabled"),
        Output("kl-weight-slider", "disabled"),
        Output("learning-rate-slider", "disabled"),
        Output("num-epochs-input", "disabled"),
        Output("latent-store", "data"),
        Output("training-poll", "disabled"),
        Input("training-poll", "n_intervals"),
        Input("training-control-store", "data"),
        State("latent-store", "data"),
    )
    def poll_training_status(
        _n_intervals: int,
        _control_store: Optional[Dict[str, int]],
        latent_store: Optional[Dict[str, int]],
    ) -> Tuple[object, bool, bool, bool, bool, bool, Dict[str, int], bool]:  # type: ignore[valid-type]
        latent_version = (latent_store or {}).get("version", 0)
        processed_messages = False
        while True:
            try:
                message = metrics_queue.get_nowait()
            except Empty:
                break
            processed_messages = True

            msg_type = message.get("type")
            if msg_type == "epoch_complete":
                _update_history_with_epoch(message)
                epoch = int(message.get("epoch", 0))
                target = int(message.get("target_epochs", 0))
                train_loss = message.get("train_loss")
                val_loss = message.get("val_loss")
                parts: List[str] = [f"Epoch {epoch}/{target if target else '?'}"]
                if train_loss is not None:
                    parts.append(f"train {float(train_loss):.4f}")
                if val_loss is not None:
                    parts.append(f"val {float(val_loss):.4f}")
                _append_status_message(" | ".join(parts))
            elif msg_type == "training_complete":
                _append_status_message("Training complete.")
            elif msg_type == "latent_updated":
                latent_version = max(latent_version, int(message.get("version", latent_version + 1)))
            elif msg_type == "error":
                _append_status_message(f"Error: {message.get('message', 'Unknown error')}")

        with state_lock:
            active = app_state["training"]["active"]
            status_messages = list(app_state["training"].get("status_messages", []))

        if status_messages:
            items = []
            for msg in status_messages[-MAX_STATUS_MESSAGES:]:
                class_name = "text-danger" if str(msg).lower().startswith("error:") else None
                items.append(html.Li(msg, className=class_name))
            status_children = html.Ul(items, className="mb-0 small")
        else:
            status_children = html.Span("Idle.", className="text-muted")

        controls_disabled = bool(active)
        latent_store_out = {"version": latent_version}
        interval_disabled = not active and not processed_messages and metrics_queue.empty()

        return (
            status_children,
            controls_disabled,
            controls_disabled,
            controls_disabled,
            controls_disabled,
            controls_disabled,
            latent_store_out,
            interval_disabled,
        )

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
