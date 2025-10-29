"""Configuration page callbacks for the SSVAE dashboard."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import dash
from dash import Dash, Input, Output, State, no_update
from dash.exceptions import PreventUpdate

from use_cases.dashboard.core import state as dashboard_state
from use_cases.dashboard.core.state import _append_status_message
from use_cases.dashboard.core.commands import UpdateModelConfigCommand


def register_config_callbacks(app: Dash) -> None:
    """Register all configuration page callbacks."""

    @app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Output("config-feedback", "children"),
        Input("config-save-btn", "n_clicks"),
        State("tc-batch-size", "value"),
        State("tc-max-epochs", "value"),
        State("tc-patience", "value"),
        State("tc-learning-rate", "value"),
        State("tc-encoder-type", "value"),
        State("tc-decoder-type", "value"),
        State("tc-latent-dim", "value"),
        State("tc-hidden-dims", "value"),
        State("tc-recon-weight", "value"),
        State("tc-kl-weight", "value"),
        State("tc-label-weight", "value"),
        State("tc-weight-decay", "value"),
        State("tc-dropout-rate", "value"),
        State("tc-grad-clip-norm", "value"),
        State("tc-monitor-metric", "value"),
        State("tc-use-contrastive", "value"),
        State("tc-contrastive-weight", "value"),
        State("training-config-store", "data"),
        prevent_initial_call=True,
    )
    def save_training_config(
        n_clicks: int,
        batch_size: Optional[int],
        max_epochs: Optional[int],
        patience: Optional[int],
        learning_rate: Optional[float],
        encoder_type: Optional[str],
        decoder_type: Optional[str],
        latent_dim: Optional[int],
        hidden_dims: Optional[str],
        recon_weight: Optional[float],
        kl_weight: Optional[float],
        label_weight: Optional[float],
        weight_decay: Optional[float],
        dropout_rate: Optional[float],
        grad_clip_norm: Optional[float],
        monitor_metric: Optional[str],
        use_contrastive: list,
        contrastive_weight: Optional[float],
        config_store: Optional[Dict],
    ) -> Tuple[str, object]:
        """Save training configuration and navigate back to main dashboard."""
        if not n_clicks:
            raise PreventUpdate
        
        model_id = None
        if isinstance(config_store, dict):
            model_id = config_store.get("_model_id")
        
        if not model_id:
            with dashboard_state.state_lock:
                if dashboard_state.app_state.active_model:
                    model_id = dashboard_state.app_state.active_model.model_id
        
        redirect_path = f"/model/{model_id}" if model_id else "/"

        command = UpdateModelConfigCommand(
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            learning_rate=learning_rate,
            encoder_type=encoder_type,
            decoder_type=decoder_type,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            recon_weight=recon_weight,
            kl_weight=kl_weight,
            label_weight=label_weight,
            weight_decay=weight_decay,
            dropout_rate=dropout_rate,
            grad_clip_norm=grad_clip_norm,
            monitor_metric=monitor_metric,
            use_contrastive=use_contrastive,
            contrastive_weight=contrastive_weight,
        )
        
        success, message = dashboard_state.dispatcher.execute(command)
        if not success:
            error_msg = dash.html.Div(
                message,
                style={"color": "#C10A27", "fontWeight": "600"},
            )
            _append_status_message(f"Configuration update failed: {message}")
            return no_update, error_msg
        
        _append_status_message(message)
        return redirect_path, ""
