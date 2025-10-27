"""Training-related callbacks for the SSVAE dashboard."""

from __future__ import annotations

import threading
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import dash
from dash import Dash, Input, Output, State, html, no_update
from dash.exceptions import PreventUpdate
import numpy as np
from queue import Empty

from training.interactive_trainer import InteractiveTrainer

from use_cases.dashboard.dashboard_callback import DashboardMetricsCallback
from use_cases.dashboard import state as dashboard_state
from use_cases.dashboard.state import (
    CHECKPOINT_PATH,
    MAX_STATUS_MESSAGES,
    metrics_queue,
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
        with dashboard_state.state_lock:
            trainer: InteractiveTrainer = dashboard_state.app_state.trainer
            x_train_ref = dashboard_state.app_state.data.x_train
            labels_ref = dashboard_state.app_state.data.labels
            target_epochs = int(dashboard_state.app_state.training.target_epochs or num_epochs)
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

        with dashboard_state.state_lock:
            model = dashboard_state.app_state.model
        latent, recon, pred_classes, pred_certainty = model.predict(x_train)

        with dashboard_state.state_lock:
            labels_latest = np.array(dashboard_state.app_state.data.labels, copy=True)
            true_labels = dashboard_state.app_state.data.true_labels
        hover_metadata = _build_hover_metadata(pred_classes, pred_certainty, labels_latest, true_labels)

        with dashboard_state.state_lock:
            dashboard_state.app_state = dashboard_state.app_state.with_training_complete(
                latent=latent,
                reconstructed=recon,
                pred_classes=pred_classes,
                pred_certainty=pred_certainty,
                hover_metadata=hover_metadata
            )
            latent_version = dashboard_state.app_state.data.version
        metrics_queue.put({"type": "latent_updated", "version": latent_version})
        metrics_queue.put({"type": "training_complete", "history": history})
    except Exception as exc:  # pragma: no cover - defensive
        _append_status_message(f"Training error: {exc}")
        metrics_queue.put({"type": "error", "message": str(exc)})
    finally:
        with dashboard_state.state_lock:
            dashboard_state.app_state = replace(
                dashboard_state.app_state,
                training=dashboard_state.app_state.training.with_idle()
            )


def register_training_callbacks(app: Dash) -> None:
    """Register all training-related callbacks."""

    @app.callback(
        Output("training-confirm-modal", "is_open"),
        Output("modal-training-info", "children"),
        Input("start-training-button", "n_clicks"),
        Input("modal-confirm-button", "n_clicks"),
        Input("modal-cancel-button", "n_clicks"),
        State("training-confirm-modal", "is_open"),
        State("num-epochs-input", "value"),
        State("recon-weight-slider", "value"),
        State("kl-weight-slider", "value"),
        State("learning-rate-slider", "value"),
        prevent_initial_call=True,
    )
    def toggle_modal(
        start_clicks: int,
        confirm_clicks: int,
        cancel_clicks: int,
        is_open: bool,
        num_epochs: Optional[float],
        recon_weight: float,
        kl_weight: float,
        learning_rate: float,
    ) -> Tuple[bool, object]:
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        # Cancel button closes modal
        if triggered_id == "modal-cancel-button":
            return False, no_update
        
        # Start button opens modal with training info
        if triggered_id == "start-training-button":
            if num_epochs is None:
                _append_status_message("Please specify the number of epochs before starting training.")
                return False, no_update
            
            try:
                epochs = int(num_epochs)
            except (TypeError, ValueError):
                _append_status_message("Epochs must be a whole number between 1 and 200.")
                return False, no_update
            
            if epochs <= 0:
                _append_status_message("Epochs must be greater than zero.")
                return False, no_update
            
            epochs = max(1, min(epochs, 200))
            
            # Estimate training time (rough: 30 seconds per epoch)
            estimated_minutes = (epochs * 30) / 60
            eta_text = f"~{int(estimated_minutes)} min" if estimated_minutes >= 1 else "<1 min"
            
            with dashboard_state.state_lock:
                labels = np.array(dashboard_state.app_state.data.labels)
                labeled_count = int(np.sum(~np.isnan(labels)))
            
            info_text = [
                html.Div(f"Train for {epochs} epoch(s) (estimated: {eta_text})", style={
                    "fontWeight": "600",
                    "marginBottom": "8px",
                }),
                html.Div([
                    html.Div(f"• Labeled samples: {labeled_count:,}"),
                    html.Div(f"• Learning rate: {learning_rate:.4f}"),
                    html.Div(f"• Reconstruction weight: {recon_weight:.0f}"),
                    html.Div(f"• KL weight: {kl_weight:.2f}"),
                ], style={
                    "fontSize": "14px",
                    "color": "#6F6F6F",
                    "lineHeight": "1.8",
                }),
            ]
            
            return True, info_text
        
        # Confirm button starts training and closes modal
        if triggered_id == "modal-confirm-button":
            return False, no_update
        
        raise PreventUpdate

    @app.callback(
        Output("training-control-store", "data"),
        Input("modal-confirm-button", "n_clicks"),
        State("recon-weight-slider", "value"),
        State("kl-weight-slider", "value"),
        State("learning-rate-slider", "value"),
        State("num-epochs-input", "value"),
        State("training-control-store", "data"),
        prevent_initial_call=True,
    )
    def handle_training_confirmation(
        confirm_clicks: int,
        recon_weight: float,
        kl_weight: float,
        learning_rate: float,
        num_epochs: Optional[float],
        control_store: Optional[Dict[str, int]],
    ) -> object:

        if not confirm_clicks:
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
            with dashboard_state.state_lock:
                if dashboard_state.app_state.training.is_active():
                    _append_status_message_locked("Training already in progress.")
                    return dash.no_update

                # Update config (mutable for now)
                dashboard_state.app_state.config.recon_weight = float(recon_weight)
                dashboard_state.app_state.config.kl_weight = float(kl_weight)
                dashboard_state.app_state.config.learning_rate = float(learning_rate)

                # Update model config
                dashboard_state.app_state.model.config.recon_weight = float(recon_weight)
                dashboard_state.app_state.model.config.kl_weight = float(kl_weight)
                dashboard_state.app_state.model.config.learning_rate = float(learning_rate)

                # Update trainer config
                dashboard_state.app_state.trainer.config.recon_weight = float(recon_weight)
                dashboard_state.app_state.trainer.config.kl_weight = float(kl_weight)
                dashboard_state.app_state.trainer.config.learning_rate = float(learning_rate)

                # Atomic training state update
                dashboard_state.app_state = replace(
                    dashboard_state.app_state,
                    training=dashboard_state.app_state.training.with_queued(epochs)
                )

            _clear_metrics_queue()
            worker = threading.Thread(target=train_worker, args=(epochs,), daemon=True)
            with dashboard_state.state_lock:
                dashboard_state.app_state = replace(
                    dashboard_state.app_state,
                    training=dashboard_state.app_state.training.with_running(worker)
                )
            worker.start()
        except Exception as exc:
            with dashboard_state.state_lock:
                dashboard_state.app_state = replace(
                    dashboard_state.app_state,
                    training=dashboard_state.app_state.training.with_idle()
                )
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

        with dashboard_state.state_lock:
            active = dashboard_state.app_state.training.is_active()
            status_messages = list(dashboard_state.app_state.training.status_messages)

        latest_messages = tuple(str(msg) for msg in status_messages[-MAX_STATUS_MESSAGES:])
        status_changed = (
            _LAST_POLL_STATE["status_messages"] is None
            or processed_messages
            or latest_messages != _LAST_POLL_STATE["status_messages"]
        )

        if status_messages:
            items = []
            # Auto-scroll to bottom by reversing and showing last N
            display_messages = status_messages[-MAX_STATUS_MESSAGES:]
            for i, msg in enumerate(display_messages):
                msg_str = str(msg)
                # Bold the last message if it's an error
                is_last = (i == len(display_messages) - 1)
                is_error = msg_str.lower().startswith("error:")
                
                style = {
                    "fontSize": "13px",
                    "fontFamily": "ui-monospace, monospace",
                    "lineHeight": "1.6",
                    "marginBottom": "4px",
                }
                
                if is_error:
                    style["color"] = "#C10A27"
                    style["fontWeight"] = "600"
                elif is_last and not msg_str.lower().startswith("idle"):
                    style["color"] = "#000000"
                    style["fontWeight"] = "500"
                else:
                    style["color"] = "#6F6F6F"
                
                items.append(html.Div(msg, style=style))
            
            status_children = html.Div(items)
        else:
            status_children = html.Div("Ready to train", style={
                "fontSize": "13px",
                "color": "#6F6F6F",
                "fontFamily": "ui-monospace, monospace",
            })

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