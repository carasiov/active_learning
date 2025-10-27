"""Configuration page callbacks for the SSVAE dashboard."""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Tuple

import dash
from dash import Dash, Input, Output, State, no_update
from dash.exceptions import PreventUpdate

from use_cases.dashboard.state import app_state, state_lock, _append_status_message


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
    ) -> Tuple[str, object]:
        """Save training configuration and navigate back to main dashboard."""
        if not n_clicks:
            raise PreventUpdate

        # Validate inputs
        errors = []

        if batch_size is None or batch_size < 32 or batch_size > 2048:
            errors.append("Batch size must be between 32 and 2048")
        if max_epochs is None or max_epochs < 1 or max_epochs > 500:
            errors.append("Max epochs must be between 1 and 500")
        if patience is None or patience < 1 or patience > 100:
            errors.append("Patience must be between 1 and 100")
        if learning_rate is None or learning_rate <= 0 or learning_rate > 0.1:
            errors.append("Learning rate must be between 0.00001 and 0.1")
        if encoder_type not in ["dense", "conv"]:
            errors.append("Encoder type must be 'dense' or 'conv'")
        if decoder_type not in ["dense", "conv"]:
            errors.append("Decoder type must be 'dense' or 'conv'")
        if latent_dim is None or latent_dim < 2:
            errors.append("Latent dimension must be at least 2")
        if hidden_dims is None or not hidden_dims.strip():
            errors.append("Hidden dimensions must be specified")
        if recon_weight is None or recon_weight < 0:
            errors.append("Reconstruction weight must be non-negative")
        if kl_weight is None or kl_weight < 0:
            errors.append("KL weight must be non-negative")
        if label_weight is None or label_weight < 0:
            errors.append("Label weight must be non-negative")
        if weight_decay is None or weight_decay < 0:
            errors.append("Weight decay must be non-negative")
        if dropout_rate is None or dropout_rate < 0 or dropout_rate > 0.5:
            errors.append("Dropout rate must be between 0.0 and 0.5")
        if grad_clip_norm is not None and grad_clip_norm < 0:
            errors.append("Gradient clip norm must be non-negative or None")
        if monitor_metric not in ["loss", "classification_loss"]:
            errors.append("Monitor metric must be 'loss' or 'classification_loss'")
        if contrastive_weight is None or contrastive_weight < 0:
            errors.append("Contrastive weight must be non-negative")

        if errors:
            error_msg = dash.html.Div(
                "; ".join(errors),
                style={"color": "#C10A27", "fontWeight": "600"},
            )
            _append_status_message(f"Configuration validation failed: {'; '.join(errors)}")
            return no_update, error_msg

        # Parse hidden_dims
        try:
            hidden_dims_tuple = tuple(int(x.strip()) for x in hidden_dims.split(","))
            if not hidden_dims_tuple or any(d <= 0 for d in hidden_dims_tuple):
                raise ValueError("Invalid hidden dimensions")
        except (ValueError, AttributeError):
            error_msg = dash.html.Div(
                "Hidden dimensions must be comma-separated positive integers (e.g., '256,128,64')",
                style={"color": "#C10A27", "fontWeight": "600"},
            )
            _append_status_message("Invalid hidden dimensions format")
            return no_update, error_msg

        # Convert grad_clip_norm: 0 means disabled (None)
        grad_clip_norm_final = None if grad_clip_norm == 0 else grad_clip_norm

        # Convert use_contrastive checkbox list to bool
        use_contrastive_bool = bool(use_contrastive)

        # Check for architecture changes
        with state_lock:
            current_config = app_state["config"]
            architecture_changed = (
                current_config.encoder_type != encoder_type
                or current_config.decoder_type != decoder_type
                or current_config.latent_dim != latent_dim
                or current_config.hidden_dims != hidden_dims_tuple
            )

        # Update app_state config
        try:
            with state_lock:
                config = app_state["config"]
                config.batch_size = int(batch_size)
                config.max_epochs = int(max_epochs)
                config.patience = int(patience)
                config.learning_rate = float(learning_rate)
                config.encoder_type = str(encoder_type)
                config.decoder_type = str(decoder_type)
                config.latent_dim = int(latent_dim)
                config.hidden_dims = hidden_dims_tuple
                config.recon_weight = float(recon_weight)
                config.kl_weight = float(kl_weight)
                config.label_weight = float(label_weight)
                config.weight_decay = float(weight_decay)
                config.dropout_rate = float(dropout_rate)
                config.grad_clip_norm = grad_clip_norm_final
                config.monitor_metric = str(monitor_metric)
                config.use_contrastive = use_contrastive_bool
                config.contrastive_weight = float(contrastive_weight)

                # Sync to model and trainer configs (for non-architecture params)
                model = app_state["model"]
                model.config.batch_size = config.batch_size
                model.config.learning_rate = config.learning_rate
                model.config.recon_weight = config.recon_weight
                model.config.kl_weight = config.kl_weight
                model.config.label_weight = config.label_weight
                model.config.weight_decay = config.weight_decay
                model.config.dropout_rate = config.dropout_rate
                model.config.grad_clip_norm = config.grad_clip_norm
                model.config.use_contrastive = config.use_contrastive
                model.config.contrastive_weight = config.contrastive_weight

                trainer = app_state["trainer"]
                trainer.config.batch_size = config.batch_size
                trainer.config.learning_rate = config.learning_rate
                trainer.config.recon_weight = config.recon_weight
                trainer.config.kl_weight = config.kl_weight
                trainer.config.label_weight = config.label_weight
                trainer.config.weight_decay = config.weight_decay
                trainer.config.dropout_rate = config.dropout_rate
                trainer.config.grad_clip_norm = config.grad_clip_norm
                trainer.config.use_contrastive = config.use_contrastive
                trainer.config.contrastive_weight = config.contrastive_weight

            if architecture_changed:
                _append_status_message(
                    "Configuration saved. Architecture changes require restarting the dashboard."
                )
            else:
                _append_status_message("Configuration updated successfully.")

        except Exception as exc:
            error_msg = dash.html.Div(
                f"Error saving configuration: {exc}",
                style={"color": "#C10A27", "fontWeight": "600"},
            )
            _append_status_message(f"Error saving configuration: {exc}")
            return no_update, error_msg

        # Navigate back to main dashboard
        return "/", no_update
