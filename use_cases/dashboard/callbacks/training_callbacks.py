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

from use_cases.dashboard.utils.training_callback import DashboardMetricsCallback
from use_cases.dashboard.core import state as dashboard_state
from use_cases.dashboard.core.state import (
    MAX_STATUS_MESSAGES,
    metrics_queue,
    _append_status_message,
    _append_status_message_locked,
    _clear_metrics_queue,
    _update_history_with_epoch,
)
from use_cases.dashboard.core.commands import StartTrainingCommand, CompleteTrainingCommand
from use_cases.dashboard.utils.visualization import _build_hover_metadata


# Performance optimization: Cache last poll state to avoid unnecessary Dash updates
# when training status hasn't changed. Critical for the 2-second polling interval.
# Trade-off: Global state (acceptable for single-user localhost app)
_LAST_POLL_STATE: Dict[str, object] = {
    "status_messages": None,
    "controls_disabled": None,
    "interval_disabled": None,
    "latent_version": None,
}


def _configure_trainer_callbacks(trainer: InteractiveTrainer, target_epochs: int, checkpoint_path: str) -> None:
    existing = getattr(trainer, "_callbacks", None)
    if existing:
        base_callbacks = [cb for cb in existing if not isinstance(cb, DashboardMetricsCallback)]
    else:
        base_callbacks = list(
            trainer.model._build_callbacks(
                weights_path=trainer.model.weights_path or checkpoint_path,
                export_history=False,
            )
        )
    base_callbacks.append(DashboardMetricsCallback(metrics_queue, target_epochs))
    trainer._callbacks = base_callbacks


def train_worker(num_epochs: int) -> None:
    """Background worker that runs incremental training and pushes updates to the UI."""

    try:
        with dashboard_state.state_lock:
            if dashboard_state.app_state.active_model is None:
                metrics_queue.put({"type": "error", "message": "No model loaded."})
                return
            
            trainer: InteractiveTrainer = dashboard_state.app_state.active_model.trainer
            x_train_ref = dashboard_state.app_state.active_model.data.x_train
            labels_ref = dashboard_state.app_state.active_model.data.labels
            target_epochs = int(dashboard_state.app_state.active_model.training.target_epochs or num_epochs)
            model_id = dashboard_state.app_state.active_model.model_id
        
        # Get model-specific checkpoint path
        from use_cases.dashboard.core.model_manager import ModelManager
        checkpoint_path = str(ModelManager.checkpoint_path(model_id))
        
        with dashboard_state.state_lock:
            if dashboard_state.app_state.active_model is None:
                return
            _configure_trainer_callbacks(trainer, target_epochs, checkpoint_path)

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
            weights_path=checkpoint_path,
        )

        with dashboard_state.state_lock:
            if dashboard_state.app_state.active_model is None:
                return
            model = dashboard_state.app_state.active_model.model
        latent, recon, pred_classes, pred_certainty = model.predict(x_train)

        with dashboard_state.state_lock:
            if dashboard_state.app_state.active_model is None:
                return
            labels_latest = np.array(dashboard_state.app_state.active_model.data.labels, copy=True)
            true_labels = dashboard_state.app_state.active_model.data.true_labels
        hover_metadata = _build_hover_metadata(pred_classes, pred_certainty, labels_latest, true_labels)

        # Use command to update state with training results
        command = CompleteTrainingCommand(
            latent=latent,
            reconstructed=recon,
            pred_classes=pred_classes,
            pred_certainty=pred_certainty,
            hover_metadata=hover_metadata
        )
        success, message = dashboard_state.dispatcher.execute(command)

        with dashboard_state.state_lock:
            if dashboard_state.app_state.active_model:
                latent_version = dashboard_state.app_state.active_model.data.version
            else:
                latent_version = 0
        metrics_queue.put({"type": "latent_updated", "version": latent_version})
        metrics_queue.put({"type": "training_complete", "history": history})
    except Exception as exc:  # pragma: no cover - defensive
        # Check if this is a user-initiated stop
        from use_cases.dashboard.utils.training_callback import TrainingStoppedException
        if isinstance(exc, TrainingStoppedException):
            _append_status_message("Training stopped by user.")
            # Still update predictions with current state
            try:
                with dashboard_state.state_lock:
                    if dashboard_state.app_state.active_model is None:
                        return
                    model = dashboard_state.app_state.active_model.model
                    x_train_ref = dashboard_state.app_state.active_model.data.x_train
                x_train = np.array(x_train_ref)
                latent, recon, pred_classes, pred_certainty = model.predict(x_train)
                
                with dashboard_state.state_lock:
                    if dashboard_state.app_state.active_model is None:
                        return
                    labels_latest = np.array(dashboard_state.app_state.active_model.data.labels, copy=True)
                    true_labels = dashboard_state.app_state.active_model.data.true_labels
                hover_metadata = _build_hover_metadata(pred_classes, pred_certainty, labels_latest, true_labels)
                
                command = CompleteTrainingCommand(
                    latent=latent,
                    reconstructed=recon,
                    pred_classes=pred_classes,
                    pred_certainty=pred_certainty,
                    hover_metadata=hover_metadata
                )
                dashboard_state.dispatcher.execute(command)
                
                with dashboard_state.state_lock:
                    if dashboard_state.app_state.active_model:
                        latent_version = dashboard_state.app_state.active_model.data.version
                        metrics_queue.put({"type": "latent_updated", "version": latent_version})
            except Exception:
                pass  # If update fails after stop, just continue
        else:
            _append_status_message(f"Training error: {exc}")
            metrics_queue.put({"type": "error", "message": str(exc)})
    finally:
        with dashboard_state.state_lock:
            if dashboard_state.app_state.active_model:
                updated_model = dashboard_state.app_state.active_model.with_training(
                    state=dashboard_state.app_state.active_model.training.state.__class__.IDLE,
                    target_epochs=0,
                    thread=None,
                    stop_requested=False
                )
                dashboard_state.app_state = dashboard_state.app_state.with_active_model(updated_model)


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
        prevent_initial_call=True,
    )
    def toggle_modal(
        start_clicks: int,
        confirm_clicks: int,
        cancel_clicks: int,
        is_open: bool,
        num_epochs: Optional[float],
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
            _append_status_message(f"Train button clicked. Epochs: {num_epochs}")
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
                if dashboard_state.app_state.active_model is None:
                    return False, html.Div("No model loaded", style={"color": "#C10A27"})
                labels = np.array(dashboard_state.app_state.active_model.data.labels)
                labeled_count = int(np.sum(~np.isnan(labels)))
                # Get current config values for display
                config = dashboard_state.app_state.active_model.config
                learning_rate = config.learning_rate
                recon_weight = config.recon_weight
                kl_weight = config.kl_weight
            
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
        State("num-epochs-input", "value"),
        State("training-control-store", "data"),
        prevent_initial_call=True,
    )
    def handle_training_confirmation(
        confirm_clicks: int,
        num_epochs: Optional[float],
        control_store: Optional[Dict[str, int]],
    ) -> object:
        if not confirm_clicks:
            raise PreventUpdate

        # Basic input validation
        if num_epochs is None:
            _append_status_message("Please specify the number of epochs before starting training.")
            return dash.no_update

        try:
            epochs = int(num_epochs)
        except (TypeError, ValueError):
            _append_status_message("Epochs must be a whole number between 1 and 200.")
            return dash.no_update

        epochs = max(1, min(epochs, 200))

        # Get current config from state
        with dashboard_state.state_lock:
            if dashboard_state.app_state.active_model is None:
                _append_status_message("No model loaded")
                return dash.no_update
            config = dashboard_state.app_state.active_model.config
            recon_weight = float(config.recon_weight)
            kl_weight = float(config.kl_weight)
            learning_rate = float(config.learning_rate)

        # Create and execute command
        command = StartTrainingCommand(
            num_epochs=epochs,
            recon_weight=recon_weight,
            kl_weight=kl_weight,
            learning_rate=learning_rate
        )

        success, message = dashboard_state.dispatcher.execute(command)

        if not success:
            _append_status_message(message)
            return dash.no_update

        try:
            _clear_metrics_queue()
            worker = threading.Thread(target=train_worker, args=(epochs,), daemon=True)
            with dashboard_state.state_lock:
                if dashboard_state.app_state.active_model:
                    updated_model = dashboard_state.app_state.active_model.with_training(
                        thread=worker
                    )
                    # Keep state as RUNNING (set by StartTrainingCommand)
                    dashboard_state.app_state = dashboard_state.app_state.with_active_model(updated_model)
            worker.start()
            _append_status_message(message)
        except Exception as exc:
            with dashboard_state.state_lock:
                if dashboard_state.app_state.active_model:
                    from use_cases.dashboard.core.state_models import TrainingState
                    updated_model = dashboard_state.app_state.active_model.with_training(
                        state=TrainingState.IDLE,
                        target_epochs=0,
                        thread=None,
                        stop_requested=False
                    )
                    dashboard_state.app_state = dashboard_state.app_state.with_active_model(updated_model)
            _append_status_message(f"Error starting training: {exc}")
            return dash.no_update

        token = (control_store or {}).get("token", 0) + 1
        return {"token": token}

    @app.callback(
        Output("training-status", "children", allow_duplicate=True),
        Output("start-training-button", "disabled", allow_duplicate=True),
        Output("num-epochs-input", "disabled", allow_duplicate=True),
        Output("latent-store", "data", allow_duplicate=True),
        Output("training-poll", "disabled", allow_duplicate=True),
        Input("training-poll", "n_intervals"),
        Input("training-control-store", "data"),
        State("latent-store", "data"),
        prevent_initial_call=True,
    )
    def poll_training_status(
        _n_intervals: int,
        _control_store: Optional[Dict[str, int]],
        latent_store: Optional[Dict[str, int]],
    ) -> Tuple[object, bool, bool, Dict[str, int], bool]:  # type: ignore[valid-type]
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
                # CompleteTrainingCommand already logged this in TrainingStatus
                pass
            elif msg_type == "latent_updated":
                latent_version = max(latent_version, int(message.get("version", latent_version + 1)))
            elif msg_type == "error":
                _append_status_message(f"Error: {message.get('message', 'Unknown error')}")

        with dashboard_state.state_lock:
            if dashboard_state.app_state.active_model:
                active = dashboard_state.app_state.active_model.training.is_active()
                status_messages = list(dashboard_state.app_state.active_model.training.status_messages)
                state_latent_version = int(dashboard_state.app_state.active_model.data.version)
            else:
                active = False
                status_messages = ["No model loaded"]
                state_latent_version = int(latent_version)

        # Ensure we pick up latent updates even if another poll drained the queue
        latent_version = max(int(latent_version), state_latent_version)

        latest_messages = tuple(str(msg) for msg in status_messages[-MAX_STATUS_MESSAGES:])
        status_changed = (
            _LAST_POLL_STATE["status_messages"] is None
            or processed_messages  # Always update if we processed messages from queue
            or latest_messages != _LAST_POLL_STATE["status_messages"]
            or active  # Always update during active training to show latest status
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
            latent_output,
            interval_output,
        )
