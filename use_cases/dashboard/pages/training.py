"""Training configuration page for the SSVAE dashboard."""

from __future__ import annotations

from typing import Dict, Optional

from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc


def build_training_config_page(model_id: str | None = None) -> html.Div:
    """Build the dedicated training configuration page."""
    dashboard_url = f"/model/{model_id}" if model_id else "/"

    return html.Div(
        [
            # Header
            html.Div(
                    [
                        dcc.Link(
                            "← Back to Dashboard",
                            href=dashboard_url,
                            style={
                                "fontSize": "14px",
                                "color": "#C10A27",
                                "textDecoration": "none",
                            "fontWeight": "600",
                            "fontFamily": "'Open Sans', Verdana, sans-serif",
                            "display": "inline-block",
                            "marginBottom": "16px",
                        },
                    ),
                    html.H1(
                        "Training Configuration",
                        style={
                            "fontSize": "28px",
                            "fontWeight": "700",
                            "color": "#000000",
                            "marginBottom": "8px",
                            "fontFamily": "'Open Sans', Verdana, sans-serif",
                        },
                    ),
                    html.P(
                        "Configure SSVAE hyperparameters for training. Changes take effect on the next training run.",
                        style={
                            "fontSize": "15px",
                            "color": "#6F6F6F",
                            "marginBottom": "32px",
                            "fontFamily": "'Open Sans', Verdana, sans-serif",
                        },
                    ),
                ],
                style={
                    "padding": "32px 48px 24px 48px",
                    "backgroundColor": "#ffffff",
                    "borderBottom": "4px solid #C10A27",
                },
            ),
            # Form content
            html.Div(
                [
                    _build_core_config_section(),
                    _build_architecture_section(),
                    _build_loss_weights_section(),
                    _build_regularization_section(),
                    _build_advanced_section(),
                    # Action buttons
                    html.Div(
                        [
                            dbc.Button(
                                "Cancel",
                                id="config-cancel-btn",
                                href=dashboard_url,
                                style={
                                    "backgroundColor": "#ffffff",
                                    "color": "#6F6F6F",
                                    "border": "1px solid #C6C6C6",
                                    "borderRadius": "8px",
                                    "padding": "12px 32px",
                                    "fontSize": "15px",
                                    "fontWeight": "600",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                    "marginRight": "16px",
                                },
                            ),
                            dbc.Button(
                                "Save Configuration",
                                id="config-save-btn",
                                n_clicks=0,
                                style={
                                    "backgroundColor": "#C10A27",
                                    "color": "#ffffff",
                                    "border": "none",
                                    "borderRadius": "8px",
                                    "padding": "12px 32px",
                                    "fontSize": "15px",
                                    "fontWeight": "700",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                        ],
                        style={
                            "display": "flex",
                            "justifyContent": "flex-end",
                            "marginTop": "32px",
                            "paddingTop": "32px",
                            "borderTop": "1px solid #C6C6C6",
                        },
                    ),
                    # Feedback area
                    html.Div(
                        id="config-feedback",
                        style={
                            "marginTop": "16px",
                            "fontSize": "14px",
                            "textAlign": "center",
                            "minHeight": "24px",
                            "fontFamily": "'Open Sans', Verdana, sans-serif",
                        },
                    ),
                ],
                style={
                    "maxWidth": "900px",
                    "margin": "0 auto",
                    "padding": "48px 48px 96px 48px",
                    "backgroundColor": "#ffffff",
                },
            ),
        ],
        style={
            "fontFamily": "'Open Sans', Verdana, sans-serif",
            "backgroundColor": "#fafafa",
            "minHeight": "100%",
            "height": "auto",
            "overflowY": "auto",
        },
    )


def _build_core_config_section() -> html.Div:
    """Build the core training parameters section."""
    return html.Div(
        [
            html.H2(
                "Core Training Parameters",
                style={
                    "fontSize": "18px",
                    "fontWeight": "700",
                    "color": "#000000",
                    "marginBottom": "20px",
                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                },
            ),
            html.Div(
                [
                    # Batch Size
                    html.Div(
                        [
                            html.Label(
                                "Batch Size",
                                style={
                                    "fontSize": "14px",
                                    "color": "#000000",
                                    "display": "block",
                                    "marginBottom": "6px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                    "fontWeight": "600",
                                },
                            ),
                            html.P(
                                "Number of samples per training batch. Larger batches are faster but require more memory.",
                                style={
                                    "fontSize": "13px",
                                    "color": "#6F6F6F",
                                    "marginBottom": "8px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                            dcc.Input(
                                id="tc-batch-size",
                                type="number",
                                min=32,
                                max=2048,
                                step=32,
                                placeholder="e.g., 256",
                                debounce=True,
                                style={
                                    "width": "100%",
                                    "padding": "10px 12px",
                                    "fontSize": "14px",
                                    "border": "1px solid #C6C6C6",
                                    "borderRadius": "6px",
                                    "fontFamily": "ui-monospace, monospace",
                                },
                            ),
                        ],
                        style={"marginBottom": "24px"},
                    ),
                    # Max Epochs
                    html.Div(
                        [
                            html.Label(
                                "Maximum Epochs",
                                style={
                                    "fontSize": "14px",
                                    "color": "#000000",
                                    "display": "block",
                                    "marginBottom": "6px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                    "fontWeight": "600",
                                },
                            ),
                            html.P(
                                "Maximum training epochs before stopping (subject to early stopping).",
                                style={
                                    "fontSize": "13px",
                                    "color": "#6F6F6F",
                                    "marginBottom": "8px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                            dcc.Input(
                                id="tc-max-epochs",
                                type="number",
                                min=1,
                                max=500,
                                step=1,
                                placeholder="e.g., 200",
                                debounce=True,
                                style={
                                    "width": "100%",
                                    "padding": "10px 12px",
                                    "fontSize": "14px",
                                    "border": "1px solid #C6C6C6",
                                    "borderRadius": "6px",
                                    "fontFamily": "ui-monospace, monospace",
                                },
                            ),
                        ],
                        style={"marginBottom": "24px"},
                    ),
                    # Patience
                    html.Div(
                        [
                            html.Label(
                                "Early Stopping Patience",
                                style={
                                    "fontSize": "14px",
                                    "color": "#000000",
                                    "display": "block",
                                    "marginBottom": "6px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                    "fontWeight": "600",
                                },
                            ),
                            html.P(
                                "Number of epochs without improvement before early stopping.",
                                style={
                                    "fontSize": "13px",
                                    "color": "#6F6F6F",
                                    "marginBottom": "8px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                            dcc.Input(
                                id="tc-patience",
                                type="number",
                                min=1,
                                max=100,
                                step=1,
                                placeholder="e.g., 20",
                                debounce=True,
                                style={
                                    "width": "100%",
                                    "padding": "10px 12px",
                                    "fontSize": "14px",
                                    "border": "1px solid #C6C6C6",
                                    "borderRadius": "6px",
                                    "fontFamily": "ui-monospace, monospace",
                                },
                            ),
                        ],
                        style={"marginBottom": "24px"},
                    ),
                    # Learning Rate
                    html.Div(
                        [
                            html.Label(
                                "Learning Rate",
                                style={
                                    "fontSize": "14px",
                                    "color": "#000000",
                                    "display": "block",
                                    "marginBottom": "6px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                    "fontWeight": "600",
                                },
                            ),
                            html.P(
                                "Optimizer learning rate. Typical range: 0.0001 to 0.01.",
                                style={
                                    "fontSize": "13px",
                                    "color": "#6F6F6F",
                                    "marginBottom": "8px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                            dcc.Input(
                                id="tc-learning-rate",
                                type="number",
                                min=0.00001,
                                max=0.1,
                                step=0.0001,
                                placeholder="e.g., 0.001",
                                debounce=True,
                                style={
                                    "width": "100%",
                                    "padding": "10px 12px",
                                    "fontSize": "14px",
                                    "border": "1px solid #C6C6C6",
                                    "borderRadius": "6px",
                                    "fontFamily": "ui-monospace, monospace",
                                },
                            ),
                        ],
                        style={"marginBottom": "0"},
                    ),
                ],
            ),
        ],
        style={
            "marginBottom": "40px",
            "paddingBottom": "32px",
            "borderBottom": "1px solid #C6C6C6",
        },
    )


def _build_architecture_section() -> html.Div:
    """Build the model architecture section."""
    return html.Div(
        [
            html.H2(
                "Model Architecture",
                style={
                    "fontSize": "18px",
                    "fontWeight": "700",
                    "color": "#000000",
                    "marginBottom": "20px",
                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                },
            ),
            html.Div(
                [
                    # Encoder Type
                    html.Div(
                        [
                            html.Label(
                                "Encoder Type",
                                style={
                                    "fontSize": "14px",
                                    "color": "#000000",
                                    "display": "block",
                                    "marginBottom": "6px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                    "fontWeight": "600",
                                },
                            ),
                            html.P(
                                "Dense: fully-connected MLP. Conv: convolutional layers (better for images).",
                                style={
                                    "fontSize": "13px",
                                    "color": "#6F6F6F",
                                    "marginBottom": "12px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                            dbc.RadioItems(
                                id="tc-encoder-type",
                                options=[
                                    {"label": "Dense (MLP)", "value": "dense"},
                                    {"label": "Convolutional", "value": "conv"},
                                ],
                                inline=False,
                                style={
                                    "fontSize": "14px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                        ],
                        style={"marginBottom": "24px"},
                    ),
                    # Decoder Type
                    html.Div(
                        [
                            html.Label(
                                "Decoder Type",
                                style={
                                    "fontSize": "14px",
                                    "color": "#000000",
                                    "display": "block",
                                    "marginBottom": "6px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                    "fontWeight": "600",
                                },
                            ),
                            html.P(
                                "Should typically match encoder type for symmetry.",
                                style={
                                    "fontSize": "13px",
                                    "color": "#6F6F6F",
                                    "marginBottom": "12px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                            dbc.RadioItems(
                                id="tc-decoder-type",
                                options=[
                                    {"label": "Dense (MLP)", "value": "dense"},
                                    {"label": "Convolutional", "value": "conv"},
                                ],
                                inline=False,
                                style={
                                    "fontSize": "14px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                        ],
                        style={"marginBottom": "24px"},
                    ),
                    # Latent Dimension
                    html.Div(
                        [
                            html.Label(
                                "Latent Dimension",
                                style={
                                    "fontSize": "14px",
                                    "color": "#000000",
                                    "display": "block",
                                    "marginBottom": "6px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                    "fontWeight": "600",
                                },
                            ),
                            html.P(
                                "Dimensionality of the latent space. 2D is useful for visualization.",
                                style={
                                    "fontSize": "13px",
                                    "color": "#6F6F6F",
                                    "marginBottom": "8px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                            dcc.Dropdown(
                                id="tc-latent-dim",
                                options=[
                                    {"label": "2 (visualization)", "value": 2},
                                    {"label": "10", "value": 10},
                                    {"label": "50", "value": 50},
                                    {"label": "100", "value": 100},
                                ],
                                clearable=False,
                                style={
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                        ],
                        style={"marginBottom": "24px"},
                    ),
                    # Hidden Dims (Dense only)
                    html.Div(
                        [
                            html.Label(
                                "Hidden Dimensions (Dense only)",
                                style={
                                    "fontSize": "14px",
                                    "color": "#000000",
                                    "display": "block",
                                    "marginBottom": "6px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                    "fontWeight": "600",
                                },
                            ),
                            html.P(
                                "Layer sizes for dense encoder. Decoder mirrors in reverse.",
                                style={
                                    "fontSize": "13px",
                                    "color": "#6F6F6F",
                                    "marginBottom": "8px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                            dcc.Dropdown(
                                id="tc-hidden-dims",
                                options=[
                                    {"label": "Small (128, 64)", "value": "128,64"},
                                    {"label": "Default (256, 128, 64)", "value": "256,128,64"},
                                    {"label": "Large (512, 256, 128)", "value": "512,256,128"},
                                ],
                                clearable=False,
                                style={
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                        ],
                        style={"marginBottom": "0"},
                    ),
                ],
            ),
            # Warning for architecture changes
            html.Div(
                [
                    html.Span("⚠️ ", style={"fontSize": "16px"}),
                    html.Span(
                        "Changing architecture (encoder/decoder type, latent dim, hidden dims) requires restarting the dashboard to take effect.",
                        style={"fontSize": "13px"},
                    ),
                ],
                style={
                    "marginTop": "20px",
                    "padding": "12px 16px",
                    "backgroundColor": "#FFF9E6",
                    "border": "1px solid #F6C900",
                    "borderRadius": "6px",
                    "color": "#4A4A4A",
                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                },
            ),
        ],
        style={
            "marginBottom": "40px",
            "paddingBottom": "32px",
            "borderBottom": "1px solid #C6C6C6",
        },
    )


def _build_loss_weights_section() -> html.Div:
    """Build the loss weights section."""
    return html.Div(
        [
            html.H2(
                "Loss Weights",
                style={
                    "fontSize": "18px",
                    "fontWeight": "700",
                    "color": "#000000",
                    "marginBottom": "20px",
                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                },
            ),
            html.Div(
                [
                    # Reconstruction Weight
                    html.Div(
                        [
                            html.Label(
                                "Reconstruction Weight",
                                style={
                                    "fontSize": "14px",
                                    "color": "#000000",
                                    "display": "block",
                                    "marginBottom": "6px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                    "fontWeight": "600",
                                },
                            ),
                            html.P(
                                "Higher values prioritize better image reconstruction quality.",
                                style={
                                    "fontSize": "13px",
                                    "color": "#6F6F6F",
                                    "marginBottom": "8px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                            dcc.Input(
                                id="tc-recon-weight",
                                type="number",
                                min=0,
                                max=10000,
                                step=100,
                                placeholder="e.g., 1000",
                                debounce=True,
                                style={
                                    "width": "100%",
                                    "padding": "10px 12px",
                                    "fontSize": "14px",
                                    "border": "1px solid #C6C6C6",
                                    "borderRadius": "6px",
                                    "fontFamily": "ui-monospace, monospace",
                                },
                            ),
                        ],
                        style={"marginBottom": "24px"},
                    ),
                    # KL Weight
                    html.Div(
                        [
                            html.Label(
                                "KL Divergence Weight",
                                style={
                                    "fontSize": "14px",
                                    "color": "#000000",
                                    "display": "block",
                                    "marginBottom": "6px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                    "fontWeight": "600",
                                },
                            ),
                            html.P(
                                "Regularizes the latent space. Typical range: 0.01 to 1.0.",
                                style={
                                    "fontSize": "13px",
                                    "color": "#6F6F6F",
                                    "marginBottom": "8px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                            dcc.Input(
                                id="tc-kl-weight",
                                type="number",
                                min=0.0,
                                max=10.0,
                                step=0.01,
                                placeholder="e.g., 0.1",
                                debounce=True,
                                style={
                                    "width": "100%",
                                    "padding": "10px 12px",
                                    "fontSize": "14px",
                                    "border": "1px solid #C6C6C6",
                                    "borderRadius": "6px",
                                    "fontFamily": "ui-monospace, monospace",
                                },
                            ),
                        ],
                        style={"marginBottom": "24px"},
                    ),
                    # Label Weight
                    html.Div(
                        [
                            html.Label(
                                "Label Weight",
                                style={
                                    "fontSize": "14px",
                                    "color": "#000000",
                                    "display": "block",
                                    "marginBottom": "6px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                    "fontWeight": "600",
                                },
                            ),
                            html.P(
                                "Weight for classification loss on labeled samples.",
                                style={
                                    "fontSize": "13px",
                                    "color": "#6F6F6F",
                                    "marginBottom": "8px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                            dcc.Input(
                                id="tc-label-weight",
                                type="number",
                                min=0.0,
                                max=10.0,
                                step=0.1,
                                placeholder="e.g., 1.0",
                                debounce=True,
                                style={
                                    "width": "100%",
                                    "padding": "10px 12px",
                                    "fontSize": "14px",
                                    "border": "1px solid #C6C6C6",
                                    "borderRadius": "6px",
                                    "fontFamily": "ui-monospace, monospace",
                                },
                            ),
                        ],
                        style={"marginBottom": "0"},
                    ),
                ],
            ),
        ],
        style={
            "marginBottom": "40px",
            "paddingBottom": "32px",
            "borderBottom": "1px solid #C6C6C6",
        },
    )


def _build_regularization_section() -> html.Div:
    """Build the regularization section."""
    return html.Div(
        [
            html.H2(
                "Regularization",
                style={
                    "fontSize": "18px",
                    "fontWeight": "700",
                    "color": "#000000",
                    "marginBottom": "20px",
                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                },
            ),
            html.Div(
                [
                    # Weight Decay
                    html.Div(
                        [
                            html.Label(
                                "Weight Decay",
                                style={
                                    "fontSize": "14px",
                                    "color": "#000000",
                                    "display": "block",
                                    "marginBottom": "6px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                    "fontWeight": "600",
                                },
                            ),
                            html.P(
                                "L2 regularization penalty. Typical range: 0.0001 to 0.01.",
                                style={
                                    "fontSize": "13px",
                                    "color": "#6F6F6F",
                                    "marginBottom": "8px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                            dcc.Input(
                                id="tc-weight-decay",
                                type="number",
                                min=0.0,
                                max=0.1,
                                step=0.0001,
                                placeholder="e.g., 0.0001",
                                debounce=True,
                                style={
                                    "width": "100%",
                                    "padding": "10px 12px",
                                    "fontSize": "14px",
                                    "border": "1px solid #C6C6C6",
                                    "borderRadius": "6px",
                                    "fontFamily": "ui-monospace, monospace",
                                },
                            ),
                        ],
                        style={"marginBottom": "24px"},
                    ),
                    # Dropout Rate
                    html.Div(
                        [
                            html.Label(
                                "Dropout Rate",
                                style={
                                    "fontSize": "14px",
                                    "color": "#000000",
                                    "display": "block",
                                    "marginBottom": "6px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                    "fontWeight": "600",
                                },
                            ),
                            html.P(
                                "Dropout probability in the classifier. Range: 0.0 to 0.5.",
                                style={
                                    "fontSize": "13px",
                                    "color": "#6F6F6F",
                                    "marginBottom": "12px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                            dcc.Slider(
                                id="tc-dropout-rate",
                                min=0.0,
                                max=0.5,
                                step=0.05,
                                marks={
                                    0.0: "0.0",
                                    0.1: "0.1",
                                    0.2: "0.2",
                                    0.3: "0.3",
                                    0.4: "0.4",
                                    0.5: "0.5",
                                },
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ],
                        style={"marginBottom": "24px"},
                    ),
                    # Gradient Clip Norm
                    html.Div(
                        [
                            html.Label(
                                "Gradient Clip Norm",
                                style={
                                    "fontSize": "14px",
                                    "color": "#000000",
                                    "display": "block",
                                    "marginBottom": "6px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                    "fontWeight": "600",
                                },
                            ),
                            html.P(
                                "Maximum gradient norm. Set to 0 to disable clipping.",
                                style={
                                    "fontSize": "13px",
                                    "color": "#6F6F6F",
                                    "marginBottom": "8px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                            dcc.Input(
                                id="tc-grad-clip-norm",
                                type="number",
                                min=0.0,
                                max=10.0,
                                step=0.1,
                                placeholder="e.g., 1.0 (0 = disabled)",
                                debounce=True,
                                style={
                                    "width": "100%",
                                    "padding": "10px 12px",
                                    "fontSize": "14px",
                                    "border": "1px solid #C6C6C6",
                                    "borderRadius": "6px",
                                    "fontFamily": "ui-monospace, monospace",
                                },
                            ),
                        ],
                        style={"marginBottom": "0"},
                    ),
                ],
            ),
        ],
        style={
            "marginBottom": "40px",
            "paddingBottom": "32px",
            "borderBottom": "1px solid #C6C6C6",
        },
    )


def _build_advanced_section() -> html.Div:
    """Build the advanced options section."""
    return html.Div(
        [
            html.H2(
                "Advanced Options",
                style={
                    "fontSize": "18px",
                    "fontWeight": "700",
                    "color": "#000000",
                    "marginBottom": "20px",
                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                },
            ),
            html.Div(
                [
                    # Monitor Metric
                    html.Div(
                        [
                            html.Label(
                                "Monitor Metric",
                                style={
                                    "fontSize": "14px",
                                    "color": "#000000",
                                    "display": "block",
                                    "marginBottom": "6px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                    "fontWeight": "600",
                                },
                            ),
                            html.P(
                                "Metric used for early stopping decisions.",
                                style={
                                    "fontSize": "13px",
                                    "color": "#6F6F6F",
                                    "marginBottom": "8px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                            dcc.Dropdown(
                                id="tc-monitor-metric",
                                options=[
                                    {"label": "Total Loss", "value": "loss"},
                                    {"label": "Classification Loss", "value": "classification_loss"},
                                ],
                                clearable=False,
                                style={
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                        ],
                        style={"marginBottom": "24px"},
                    ),
                    # Use Contrastive
                    html.Div(
                        [
                            html.Label(
                                "Contrastive Learning",
                                style={
                                    "fontSize": "14px",
                                    "color": "#000000",
                                    "display": "block",
                                    "marginBottom": "6px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                    "fontWeight": "600",
                                },
                            ),
                            html.P(
                                "Enable experimental contrastive loss term.",
                                style={
                                    "fontSize": "13px",
                                    "color": "#6F6F6F",
                                    "marginBottom": "12px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                            dbc.Checklist(
                                id="tc-use-contrastive",
                                options=[{"label": "Enable contrastive loss", "value": True}],
                                value=[],
                                style={
                                    "fontSize": "14px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                        ],
                        style={"marginBottom": "24px"},
                    ),
                    # Contrastive Weight
                    html.Div(
                        [
                            html.Label(
                                "Contrastive Weight",
                                style={
                                    "fontSize": "14px",
                                    "color": "#000000",
                                    "display": "block",
                                    "marginBottom": "6px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                    "fontWeight": "600",
                                },
                            ),
                            html.P(
                                "Weight for contrastive loss (only applies if enabled above).",
                                style={
                                    "fontSize": "13px",
                                    "color": "#6F6F6F",
                                    "marginBottom": "8px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                            dcc.Input(
                                id="tc-contrastive-weight",
                                type="number",
                                min=0.0,
                                max=10.0,
                                step=0.1,
                                placeholder="e.g., 0.0",
                                debounce=True,
                                style={
                                    "width": "100%",
                                    "padding": "10px 12px",
                                    "fontSize": "14px",
                                    "border": "1px solid #C6C6C6",
                                    "borderRadius": "6px",
                                    "fontFamily": "ui-monospace, monospace",
                                },
                            ),
                        ],
                        style={"marginBottom": "0"},
                    ),
                ],
            ),
        ],
        style={
            "marginBottom": "0",
        },
    )


def register_config_page_callbacks(app):
    """Register callbacks to populate form from config store."""
    
    @app.callback(
        Output("tc-batch-size", "value"),
        Output("tc-max-epochs", "value"),
        Output("tc-patience", "value"),
        Output("tc-learning-rate", "value"),
        Output("tc-encoder-type", "value"),
        Output("tc-decoder-type", "value"),
        Output("tc-latent-dim", "value"),
        Output("tc-hidden-dims", "value"),
        Output("tc-recon-weight", "value"),
        Output("tc-kl-weight", "value"),
        Output("tc-label-weight", "value"),
        Output("tc-weight-decay", "value"),
        Output("tc-dropout-rate", "value"),
        Output("tc-grad-clip-norm", "value"),
        Output("tc-monitor-metric", "value"),
        Output("tc-use-contrastive", "value"),
        Output("tc-contrastive-weight", "value"),
        Input("training-config-store", "data"),
    )
    def populate_form_from_config(config_dict: Optional[Dict]) -> tuple:
        """Populate form fields from the config store."""
        if not config_dict:
            # Return empty/default values if no config
            return (256, 200, 20, 0.001, "dense", "dense", 2, "256,128,64", 
                   1000.0, 0.1, 1.0, 0.0001, 0.2, 1.0, "classification_loss", [], 0.0)
        
        # Convert hidden_dims tuple to comma-separated string
        hidden_dims_str = ",".join(str(d) for d in config_dict.get("hidden_dims", (256, 128, 64)))
        
        # Convert grad_clip_norm: None -> 0
        grad_clip = config_dict.get("grad_clip_norm", 1.0)
        grad_clip_value = 0.0 if grad_clip is None else grad_clip
        
        # Convert use_contrastive bool to checkbox list
        use_contrastive_list = [True] if config_dict.get("use_contrastive", False) else []
        
        return (
            config_dict.get("batch_size", 256),
            config_dict.get("max_epochs", 200),
            config_dict.get("patience", 20),
            config_dict.get("learning_rate", 0.001),
            config_dict.get("encoder_type", "dense"),
            config_dict.get("decoder_type", "dense"),
            config_dict.get("latent_dim", 2),
            hidden_dims_str,
            config_dict.get("recon_weight", 1000.0),
            config_dict.get("kl_weight", 0.1),
            config_dict.get("label_weight", 1.0),
            config_dict.get("weight_decay", 0.0001),
            config_dict.get("dropout_rate", 0.2),
            grad_clip_value,
            config_dict.get("monitor_metric", "classification_loss"),
            use_contrastive_list,
            config_dict.get("contrastive_weight", 0.0),
        )
