"""Training-related callbacks for the SSVAE dashboard."""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple

import dash
from dash import Dash, Input, Output, State, html, no_update
from dash.exceptions import PreventUpdate
import numpy as np
from queue import Empty

from training.interactive_trainer import InteractiveTrainer

from use_cases.dashboard.dashboard_callback import DashboardMetricsCallback
from use_cases.dashboard.state import (
    CHECKPOINT_PATH,
    MAX_STATUS_MESSAGES,
    app_state,
    metrics_queue,
    state_lock,
    _append_status_message,
    _append_status_message_locked,
    _clear_metrics_queue,
    _update_history_with_epoch,
)
from use_cases.dashboard.utils import _build_hover_metadata

_LAST_POLL_STATE: Dict[str, object] = {
    "status_messages": None,
    "controls_disabled": None,
    "interval_disabled": None,
    "latent_version": None,
}


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
        hover_metadata = _build_hover_metadata(pred_classes, pred_certainty, labels_latest, true_labels)

        with state_lock:
            app_state["data"]["latent"] = latent
            app_state["data"]["reconstructed"] = recon
            app_state["data"]["pred_classes"] = pred_classes
            app_state["data"]["pred_certainty"] = pred_certainty
            app_state["data"]["hover_metadata"] = hover_metadata
            app_state["ui"]["latent_version"] = int(app_state["ui"]["latent_version"]) + 1
            latent_version = app_state["ui"]["latent_version"]
        metrics_queue.put({"type": "latent_updated", "version": latent_version})
        metrics_queue.put({"type": "training_complete", "history": history})
    except Exception as exc:  # pragma: no cover - defensive
        _append_status_message(f"Training error: {exc}")
        metrics_queue.put({"type": "error", "message": str(exc)})
    finally:
        with state_lock:
            app_state["training"]["active"] = False
            app_state["training"]["thread"] = None


def register_training_callbacks(app: Dash) -> None:
    """Register all training-related callbacks."""

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

        try:
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
        except Exception as exc:
            with state_lock:
                app_state["training"]["active"] = False
                app_state["training"]["thread"] = None
            _append_status_message(f"Error starting training: {exc}")
            return dash.no_update

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

        latest_messages = tuple(str(msg) for msg in status_messages[-MAX_STATUS_MESSAGES:])
        status_changed = (
            _LAST_POLL_STATE["status_messages"] is None
            or processed_messages
            or latest_messages != _LAST_POLL_STATE["status_messages"]
        )

        if status_messages:
            items = []
            for msg in status_messages[-MAX_STATUS_MESSAGES:]:
                class_name = "text-danger" if str(msg).lower().startswith("error:") else None
                items.append(html.Li(msg, className=class_name))
            status_children = html.Ul(items, className="mb-0 small")
        else:
            status_children = html.Span("Idle.", className="text-muted")

        controls_disabled = bool(active)
        controls_changed = (
            _LAST_POLL_STATE["controls_disabled"] is None
            or controls_disabled != _LAST_POLL_STATE["controls_disabled"]
        )

        interval_disabled = not active and not processed_messages and metrics_queue.empty()
        interval_changed = (
            _LAST_POLL_STATE["interval_disabled"] is None
            or interval_disabled != _LAST_POLL_STATE["interval_disabled"]
        )

        latent_changed = (
            _LAST_POLL_STATE["latent_version"] is None
            or latent_version != _LAST_POLL_STATE["latent_version"]
        )
        latent_store_out = {"version": latent_version}

        # Update cached state with the latest values.
        _LAST_POLL_STATE["status_messages"] = latest_messages
        _LAST_POLL_STATE["controls_disabled"] = controls_disabled
        _LAST_POLL_STATE["interval_disabled"] = interval_disabled
        _LAST_POLL_STATE["latent_version"] = latent_version

        status_output = status_children if status_changed else no_update
        control_output = controls_disabled if controls_changed else no_update
        latent_output = latent_store_out if latent_changed else no_update
        interval_output = interval_disabled if interval_changed else no_update

        return (
            status_output,
            control_output,
            control_output,
            control_output,
            control_output,
            control_output,
            latent_output,
            interval_output,
        )
