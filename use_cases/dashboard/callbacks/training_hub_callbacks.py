"""Training Hub callbacks - dedicated training monitoring page."""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Tuple

import dash
from dash import Dash, Input, Output, State, html, no_update
from dash.exceptions import PreventUpdate
import numpy as np
import plotly.graph_objects as go
from queue import Empty

from rcmvae.application.runtime.interactive import InteractiveTrainer

from use_cases.dashboard.utils.training_callback import DashboardMetricsCallback
from use_cases.dashboard.core import state as dashboard_state
from use_cases.dashboard.core.state import (
    MAX_STATUS_MESSAGES,
    metrics_queue,
    _append_status_message,
    _clear_metrics_queue,
    _update_history_with_epoch,
)
from use_cases.dashboard.core.commands import StartTrainingCommand, CompleteTrainingCommand
from use_cases.dashboard.utils.visualization import (
    compute_ema_smoothing,
    _build_hover_metadata,
    INFOTEAM_PALETTE,
)


# Cache for Training Hub polling to avoid unnecessary updates
_HUB_LAST_POLL_STATE: Dict[str, object] = {
    "status_messages": None,
    "controls_disabled": None,
    "interval_disabled": None,
    "latent_version": None,
}


def _configure_trainer_callbacks(trainer: InteractiveTrainer, target_epochs: int, checkpoint_path: str) -> None:
    """Configure trainer with dashboard callback for metrics reporting."""
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


def train_worker_hub(num_epochs: int) -> None:
    """Background worker for Training Hub page - identical logic to main page worker."""
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
            run_epoch_offset = len(dashboard_state.app_state.active_model.history.epochs)
        
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
        start_time = time.perf_counter()
        history = trainer.train_epochs(
            num_epochs=target_epochs,
            data=x_train,
            labels=labels,
            weights_path=checkpoint_path,
        )
        train_time = time.perf_counter() - start_time
        epochs_completed = int(len(history.get("loss", []))) if isinstance(history, dict) else target_epochs

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
            hover_metadata=hover_metadata,
            train_time=train_time,
            epoch_offset=run_epoch_offset,
            epochs_completed=epochs_completed,
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
                    total_epochs = len(dashboard_state.app_state.active_model.history.epochs)
                hover_metadata = _build_hover_metadata(pred_classes, pred_certainty, labels_latest, true_labels)
                
                command = CompleteTrainingCommand(
                    latent=latent,
                    reconstructed=recon,
                    pred_classes=pred_classes,
                    pred_certainty=pred_certainty,
                    hover_metadata=hover_metadata,
                    epoch_offset=run_epoch_offset,
                    epochs_completed=max(0, total_epochs - run_epoch_offset),
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
                from use_cases.dashboard.core.state_models import TrainingState
                updated_model = dashboard_state.app_state.active_model.with_training(
                    state=TrainingState.IDLE,
                    target_epochs=0,
                    thread=None,
                    stop_requested=False
                )
                dashboard_state.app_state = dashboard_state.app_state.with_active_model(updated_model)


def register_training_hub_callbacks(app: Dash) -> None:
    """Register all Training Hub page callbacks."""

    @app.callback(
        Output("training-hub-loss-curves", "figure"),
        Input("training-hub-latent-store", "data"),
        Input("training-hub-loss-smoothing", "value"),
        Input("training-hub-poll", "n_intervals"),  # Update on every poll for real-time curves
    )
    def update_hub_loss_curves(_latent_store: dict | None, smoothing_enabled: list, _n_intervals: int):
        """Render training progress loss curves for Training Hub."""
        with dashboard_state.state_lock:
            if dashboard_state.app_state.active_model is None:
                # Return empty figure if no model loaded
                figure = go.Figure()
                figure.update_layout(
                    template="plotly_white",
                    xaxis_title=dict(text="Epoch", font=dict(size=14)),
                    yaxis_title=dict(text="Loss", font=dict(size=14)),
                    margin=dict(l=40, r=20, t=30, b=40),
                )
                return figure
            history = dashboard_state.app_state.active_model.history
            epochs = list(history.epochs)

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
            values = getattr(history, key, [])
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
            font=dict(size=12),
        )
        figure.update_xaxes(tickfont=dict(size=12))
        figure.update_yaxes(tickfont=dict(size=12))
        return figure

    @app.callback(
        Output("training-hub-terminal", "children"),
        Input("training-hub-control-store", "data"),
        Input("training-hub-latent-store", "data"),
        Input("training-hub-poll", "n_intervals"),  # Also update on poll to show real-time messages
    )
    def update_hub_terminal(_control: dict | None, _latent: dict | None, _n_intervals: int):
        """Show training status messages in terminal-style output."""
        with dashboard_state.state_lock:
            if dashboard_state.app_state.active_model:
                status_messages = list(dashboard_state.app_state.active_model.training.status_messages)
            else:
                status_messages = ["No model loaded"]
        
        if not status_messages:
            return [html.Div("Ready to train. Click 'Start Training' to begin.", style={
                "fontFamily": "ui-monospace, 'SF Mono', Monaco, monospace",
                "fontSize": "13px",
                "color": "#d4d4d4",
                "lineHeight": "1.6",
            })]
        
        terminal_lines = []
        display_messages = status_messages[-50:]  # Last 50 messages
        for msg in display_messages:
            msg_str = str(msg)
            is_error = msg_str.lower().startswith("error:")
            
            style = {
                "fontFamily": "ui-monospace, 'SF Mono', Monaco, monospace",
                "fontSize": "13px",
                "lineHeight": "1.6",
                "marginBottom": "2px",
            }
            
            if is_error:
                style["color"] = "#f48771"  # Light red for dark background
                style["fontWeight"] = "600"
            else:
                style["color"] = "#d4d4d4"
            
            terminal_lines.append(html.Div(msg_str, style=style))
        
        return terminal_lines

    # Auto-scroll terminal to bottom
    app.clientside_callback(
        """
        function(children) {
            setTimeout(function() {
                const terminal = document.getElementById('training-hub-terminal');
                if (terminal) {
                    terminal.scrollTop = terminal.scrollHeight;
                }
            }, 100);
            return window.dash_clientside.no_update;
        }
        """,
        Output("training-hub-terminal", "style"),
        Input("training-hub-terminal", "children"),
    )

    @app.callback(
        Output("training-hub-status-metrics", "children"),
        Input("training-hub-control-store", "data"),
        Input("training-hub-latent-store", "data"),
    )
    def update_hub_status_metrics(_control: dict | None, _latent: dict | None):
        """Show live training metrics in status hero bar."""
        with dashboard_state.state_lock:
            if dashboard_state.app_state.active_model is None:
                return "No model loaded"
            training_state = dashboard_state.app_state.active_model.training.state
            history = dashboard_state.app_state.active_model.history
            epochs = list(history.epochs)
        
        if training_state.name == "RUNNING":
            if epochs:
                last_epoch = epochs[-1]
                train_loss = history.train_loss[-1] if history.train_loss else None
                val_loss = history.val_loss[-1] if history.val_loss else None
                
                metric_parts = [f"Epoch {last_epoch}"]
                if train_loss is not None:
                    metric_parts.append(f"Train: {train_loss:.4f}")
                if val_loss is not None:
                    metric_parts.append(f"Val: {val_loss:.4f}")
                
                return " | ".join(metric_parts)
            return "Training started..."
        
        elif training_state.name == "COMPLETE":
            if epochs:
                return f"Training complete. Total epochs: {len(epochs)}"
            return "Training complete"
        
        elif training_state.name == "ERROR":
            return "Training error. Check terminal for details."
        
        else:  # IDLE
            return "Ready to train"

    @app.callback(
        Output("training-hub-params-content", "style"),
        Output("training-hub-params-toggle", "children"),
        Input("training-hub-params-header", "n_clicks"),
        State("training-hub-params-content", "style"),
        prevent_initial_call=True,
    )
    def toggle_hub_params(n_clicks: int, current_style: dict):
        """Toggle Essential Parameters collapsible section."""
        if not n_clicks:
            raise PreventUpdate
        
        is_visible = current_style.get("display", "block") == "block"
        
        if is_visible:
            new_style = {"display": "none"}
            arrow = "▶"
        else:
            new_style = {"display": "block"}
            arrow = "▼"
        
        return new_style, arrow

    @app.callback(
        Output("training-hub-terminal", "children", allow_duplicate=True),
        Input("training-hub-terminal-clear", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_hub_terminal(n_clicks: int):
        """Clear terminal output."""
        if not n_clicks:
            raise PreventUpdate
        
        # Clear status messages in state
        with dashboard_state.state_lock:
            if dashboard_state.app_state.active_model:
                from use_cases.dashboard.core.state_models import TrainingStatus, TrainingState
                updated_model = dashboard_state.app_state.active_model.with_training(
                    status_messages=[]
                )
                dashboard_state.app_state = dashboard_state.app_state.with_active_model(updated_model)
        
        return [html.Div("Terminal cleared", style={
            "fontFamily": "ui-monospace, 'SF Mono', Monaco, monospace",
            "fontSize": "13px",
            "color": "#d4d4d4",
            "lineHeight": "1.6",
        })]

    @app.callback(
        Output("training-hub-terminal-download", "href"),
        Input("training-hub-terminal-download", "n_clicks"),
        prevent_initial_call=True,
    )
    def download_hub_logs(n_clicks: int):
        """Download terminal logs as text file."""
        if not n_clicks:
            raise PreventUpdate
        
        with dashboard_state.state_lock:
            if dashboard_state.app_state.active_model:
                status_messages = list(dashboard_state.app_state.active_model.training.status_messages)
            else:
                status_messages = ["No model loaded"]
        
        if not status_messages:
            log_content = "No training logs available.\n"
        else:
            log_content = "\n".join(str(msg) for msg in status_messages)
        
        # Create data URI for download
        import urllib.parse
        encoded = urllib.parse.quote(log_content)
        data_uri = f"data:text/plain;charset=utf-8,{encoded}"
        
        return data_uri

    # Training Modal and Confirmation (duplicated from main page with Hub IDs)
    @app.callback(
        Output("training-hub-modal", "is_open"),
        Output("training-hub-modal-info", "children"),
        Input("training-hub-start-button", "n_clicks"),
        Input("training-hub-modal-confirm", "n_clicks"),
        Input("training-hub-modal-cancel", "n_clicks"),
        State("training-hub-modal", "is_open"),
        State("training-hub-epochs", "value"),
        State("training-hub-recon", "value"),
        State("training-hub-kl", "value"),
        State("training-hub-lr", "value"),
        prevent_initial_call=True,
    )
    def toggle_hub_training_modal(
        start_clicks: int,
        confirm_clicks: int,
        cancel_clicks: int,
        is_open: bool,
        num_epochs: Optional[float],
        recon_weight: Optional[float],
        kl_weight: Optional[float],
        learning_rate: Optional[float],
    ) -> Tuple[bool, object]:
        """Handle training confirmation modal for Training Hub."""
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        # Cancel button closes modal
        if triggered_id == "training-hub-modal-cancel":
            return False, no_update
        
        # Start button opens modal with training info
        if triggered_id == "training-hub-start-button":
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
                if dashboard_state.app_state.active_model:
                    labels = np.array(dashboard_state.app_state.active_model.data.labels)
                    labeled_count = int(np.sum(~np.isnan(labels)))
                else:
                    labeled_count = 0
            
            info_text = [
                html.Div(f"Train for {epochs} epoch(s) (estimated: {eta_text})", style={
                    "fontWeight": "600",
                    "marginBottom": "8px",
                }),
                html.Div([
                    html.Div(f"• Labeled samples: {labeled_count:,}"),
                    html.Div(f"• Learning rate: {learning_rate:.4f}" if learning_rate is not None else "• Learning rate: not set"),
                    html.Div(f"• Reconstruction weight: {recon_weight:.0f}" if recon_weight is not None else "• Reconstruction weight: not set"),
                    html.Div(f"• KL weight: {kl_weight:.2f}" if kl_weight is not None else "• KL weight: not set"),
                ], style={
                    "fontSize": "14px",
                    "color": "#6F6F6F",
                    "lineHeight": "1.8",
                }),
            ]
            
            return True, info_text
        
        # Confirm button starts training and closes modal
        if triggered_id == "training-hub-modal-confirm":
            return False, no_update
        
        raise PreventUpdate

    @app.callback(
        Output("training-hub-control-store", "data"),
        Input("training-hub-modal-confirm", "n_clicks"),
        State("training-hub-recon", "value"),
        State("training-hub-kl", "value"),
        State("training-hub-lr", "value"),
        State("training-hub-epochs", "value"),
        State("training-hub-control-store", "data"),
        prevent_initial_call=True,
    )
    def handle_hub_training_confirmation(
        confirm_clicks: int,
        recon_weight: Optional[float],
        kl_weight: Optional[float],
        learning_rate: Optional[float],
        num_epochs: Optional[float],
        control_store: Optional[Dict[str, int]],
    ) -> object:
        """Start training from Training Hub when user confirms."""
        if not confirm_clicks:
            raise PreventUpdate

        # Basic input validation
        if num_epochs is None:
            _append_status_message("Please specify the number of epochs before starting training.")
            return no_update

        try:
            epochs = int(num_epochs)
        except (TypeError, ValueError):
            _append_status_message("Epochs must be a whole number between 1 and 200.")
            return no_update

        epochs = max(1, min(epochs, 200))

        if recon_weight is None or kl_weight is None or learning_rate is None:
            _append_status_message("Please configure all hyperparameters before training.")
            return no_update

        # Create and execute command
        command = StartTrainingCommand(
            num_epochs=epochs,
            recon_weight=float(recon_weight),
            kl_weight=float(kl_weight),
            learning_rate=float(learning_rate)
        )

        success, message = dashboard_state.dispatcher.execute(command)

        if not success:
            _append_status_message(message)
            return no_update

        try:
            _clear_metrics_queue()
            worker = threading.Thread(target=train_worker_hub, args=(epochs,), daemon=True)
            with dashboard_state.state_lock:
                if dashboard_state.app_state.active_model:
                    updated_model = dashboard_state.app_state.active_model.with_training(
                        thread=worker
                    )
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
            return no_update

        token = (control_store or {}).get("token", 0) + 1
        return {"token": token}

    @app.callback(
        Output("training-hub-control-store", "data", allow_duplicate=True),
        Input("training-hub-stop-button", "n_clicks"),
        State("training-hub-control-store", "data"),
        prevent_initial_call=True,
    )
    def handle_hub_stop_training(n_clicks: int, control_store: dict) -> dict:
        """Handle stop training button click."""
        from use_cases.dashboard.core.commands import StopTrainingCommand
        
        command = StopTrainingCommand()
        success, message = dashboard_state.dispatcher.execute(command)
        
        if success:
            _append_status_message(message)
        else:
            _append_status_message(f"Cannot stop: {message}")
        
        token = (control_store or {}).get("token", 0) + 1
        return {"token": token}

    @app.callback(
        Output("training-hub-latent-store", "data", allow_duplicate=True),
        Output("training-hub-start-button", "disabled"),
        Output("training-hub-stop-button", "disabled"),
        Output("training-hub-epochs", "disabled"),
        Output("training-hub-lr", "disabled"),
        Output("training-hub-recon", "disabled"),
        Output("training-hub-kl", "disabled"),
        Output("training-hub-poll", "disabled"),
        Input("training-hub-poll", "n_intervals"),
        Input("training-hub-control-store", "data"),
        State("training-hub-latent-store", "data"),
        prevent_initial_call=True,
    )
    def poll_hub_training_status(
        _n_intervals: int,
        _control_store: Optional[Dict[str, int]],
        latent_store: Optional[Dict[str, int]],
    ) -> Tuple[Dict[str, int], bool, bool, bool, bool, bool, bool, bool]:
        """Poll training status and update Training Hub UI state."""
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
                # TrainingStatus already includes a completion message
                pass
            elif msg_type == "training_stopped":
                _append_status_message(message.get("message", "Training stopped."))
            elif msg_type == "latent_updated":
                latent_version = max(latent_version, int(message.get("version", latent_version + 1)))
            elif msg_type == "error":
                _append_status_message(f"Error: {message.get('message', 'Unknown error')}")

        with dashboard_state.state_lock:
            if dashboard_state.app_state.active_model:
                active = dashboard_state.app_state.active_model.training.is_active()
                state_latent_version = int(dashboard_state.app_state.active_model.data.version)
            else:
                active = False
                state_latent_version = int(latent_version)

        # Ensure hub picks up latent updates even if another poll drained the queue
        latent_version = max(int(latent_version), state_latent_version)

        controls_disabled = bool(active)
        controls_changed = (
            _HUB_LAST_POLL_STATE["controls_disabled"] is None
            or controls_disabled != _HUB_LAST_POLL_STATE["controls_disabled"]
        )

        interval_disabled = not active and not processed_messages and metrics_queue.empty()
        interval_changed = (
            _HUB_LAST_POLL_STATE["interval_disabled"] is None
            or interval_disabled != _HUB_LAST_POLL_STATE["interval_disabled"]
        )

        latent_changed = (
            _HUB_LAST_POLL_STATE["latent_version"] is None
            or latent_version != _HUB_LAST_POLL_STATE["latent_version"]
        )
        latent_store_out = {"version": latent_version}

        # Update cached state
        _HUB_LAST_POLL_STATE["controls_disabled"] = controls_disabled
        _HUB_LAST_POLL_STATE["interval_disabled"] = interval_disabled
        _HUB_LAST_POLL_STATE["latent_version"] = latent_version

        # Outputs
        latent_output = latent_store_out if latent_changed else no_update
        control_output = controls_disabled if controls_changed else no_update
        stop_disabled = not controls_disabled  # Stop button enabled when training
        interval_output = interval_disabled if interval_changed else no_update

        return (
            latent_output,
            control_output,  # start-button disabled
            stop_disabled,   # stop-button disabled (inverse)
            control_output,  # epochs disabled
            control_output,  # lr disabled
            control_output,  # recon disabled
            control_output,  # kl disabled
            interval_output, # poll disabled
        )
    
    @app.callback(
        Output("training-hub-status-text", "children"),
        Output("training-hub-status-text", "style"),
        Input("training-hub-control-store", "data"),
    )
    def update_hub_status_text(_control: dict):
        """Update status text and color based on training state."""
        with dashboard_state.state_lock:
            if dashboard_state.app_state.active_model:
                training_state = dashboard_state.app_state.active_model.training.state
            else:
                from use_cases.dashboard.core.state_models import TrainingState
                training_state = TrainingState.IDLE
        
        status_map = {
            "IDLE": ("IDLE", "#6F6F6F"),
            "QUEUED": ("QUEUED", "#F6C900"),
            "RUNNING": ("RUNNING", "#45717A"),
            "COMPLETE": ("COMPLETE", "#AFCC37"),
            "ERROR": ("ERROR", "#C10A27"),
        }
        
        text, color = status_map.get(training_state.name, ("UNKNOWN", "#6F6F6F"))
        
        style = {
            "display": "inline-block",
            "padding": "6px 16px",
            "backgroundColor": color,
            "color": "#ffffff",
            "borderRadius": "20px",
            "fontSize": "13px",
            "fontWeight": "700",
            "letterSpacing": "0.5px",
            "fontFamily": "'Open Sans', Verdana, sans-serif",
        }
        
        return text, style
