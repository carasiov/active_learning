"""Training Hub page - Dedicated training monitoring and configuration interface."""

from __future__ import annotations

import sys
from pathlib import Path

from dash import dcc, html
import dash_bootstrap_components as dbc
import numpy as np

# Ensure repository imports work
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from use_cases.dashboard.core import state as dashboard_state
from use_cases.dashboard.core.model_manager import ModelManager


def build_training_hub_layout() -> html.Div:
    """Build the Training Hub page layout."""
    dashboard_state.initialize_model_and_data()
    
    with dashboard_state.state_lock:
        # Check if we have an active model
        if dashboard_state.app_state.active_model is None:
            return html.Div([
                html.H3("No Model Loaded", style={"textAlign": "center", "marginTop": "100px"}),
                html.P("Please select a model from the home page.", style={"textAlign": "center"}),
                html.A("Go to Home", href="/", style={"display": "block", "textAlign": "center"})
            ])
        
        config = dashboard_state.app_state.active_model.config
        training_state = dashboard_state.app_state.active_model.training.state
        target_epochs = dashboard_state.app_state.active_model.training.target_epochs or 10
        status_messages = list(dashboard_state.app_state.active_model.training.status_messages)
        latent_version = dashboard_state.app_state.active_model.data.version
        model_id = dashboard_state.app_state.active_model.model_id
    
    # Determine status for hero bar
    if training_state.name == "RUNNING":
        status_color = "#45717A"  # Teal
        status_text = "RUNNING"
    elif training_state.name == "COMPLETE":
        status_color = "#AFCC37"  # Lime
        status_text = "COMPLETE"
    elif training_state.name == "ERROR":
        status_color = "#C10A27"  # Red
        status_text = "ERROR"
    else:  # IDLE or QUEUED
        status_color = "#6F6F6F"  # Gray
        status_text = "IDLE"
    
    checkpoint_path = ModelManager.checkpoint_path(model_id)
    try:
        checkpoint_display = str(checkpoint_path.relative_to(ROOT_DIR))
    except ValueError:
        checkpoint_display = str(checkpoint_path)

    return html.Div(
        [
            # Hidden stores and intervals - use session storage to persist across navigation
            dcc.Store(id="training-hub-control-store", data={"token": 0}, storage_type='session'),
            dcc.Store(id="training-hub-latent-store", data={"version": latent_version}, storage_type='session'),
            dcc.Interval(id="training-hub-poll", interval=2000, n_intervals=0, disabled=False),
            
            # Training confirmation modal
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Confirm Training")),
                    dbc.ModalBody(
                        [
                            html.Div(id="training-hub-modal-info", style={
                                "fontSize": "15px",
                                "lineHeight": "1.6",
                                "color": "#4A4A4A",
                                "fontFamily": "'Open Sans', Verdana, sans-serif",
                            }),
                            html.Div(
                                f"⚠️ This will overwrite the current checkpoint at {checkpoint_display}",
                                style={
                                    "marginTop": "16px",
                                    "padding": "12px",
                                    "backgroundColor": "#F6E3AC",
                                    "border": "1px solid #F6C900",
                                    "borderRadius": "6px",
                                    "fontSize": "14px",
                                    "color": "#4A4A4A",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                        ]
                    ),
                    dbc.ModalFooter(
                        [
                            dbc.Button(
                                "Cancel",
                                id="training-hub-modal-cancel",
                                n_clicks=0,
                                style={
                                    "backgroundColor": "#ffffff",
                                    "color": "#6F6F6F",
                                    "border": "1px solid #C6C6C6",
                                    "borderRadius": "6px",
                                    "padding": "8px 16px",
                                    "fontSize": "13px",
                                    "fontWeight": "600",
                                },
                            ),
                            dbc.Button(
                                "Start Training",
                                id="training-hub-modal-confirm",
                                n_clicks=0,
                                style={
                                    "backgroundColor": "#C10A27",
                                    "color": "#ffffff",
                                    "border": "none",
                                    "borderRadius": "6px",
                                    "padding": "8px 16px",
                                    "fontSize": "13px",
                                    "fontWeight": "700",
                                    "marginLeft": "8px",
                                },
                            ),
                        ]
                    ),
                ],
                id="training-hub-modal",
                is_open=False,
                centered=True,
            ),
            
            # Header with logo and red accent bar
            html.Div([
                html.Div(
                    [
                        html.Div(
                            [
                                html.Img(
                                    src="/assets/infoteam_logo_basic.png",
                                    alt="infoteam software",
                                    style={"height": "50px", "width": "auto", "display": "block"},
                                ),
                            ],
                            style={"display": "inline-block", "marginRight": "32px"},
                        ),
                        html.Div(
                            [
                                html.H1("Training Hub", style={
                                    "fontSize": "24px",
                                    "fontWeight": "700",
                                    "margin": "0",
                                    "color": "#000000",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                }),
                                html.Div("Monitor Training Progress & Configure Hyperparameters", style={
                                    "fontSize": "15px",
                                    "color": "#6F6F6F",
                                    "marginTop": "2px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                }),
                            ],
                            style={"display": "inline-block", "verticalAlign": "middle"},
                        ),
                        html.Div(
                            dcc.Link(
                                "← Back to Latent Viewer",
                                href=f"/model/{model_id}",
                                style={
                                    "fontSize": "14px",
                                    "color": "#45717A",
                                    "textDecoration": "none",
                                    "fontWeight": "600",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                            style={"marginLeft": "auto", "display": "inline-block"},
                        ),
                    ],
                    style={
                        "padding": "16px 32px",
                        "backgroundColor": "#ffffff",
                        "display": "flex",
                        "alignItems": "center",
                    },
                ),
                html.Div(style={
                    "height": "4px",
                    "backgroundColor": "#C10A27",
                    "width": "100%",
                }),
            ]),
            
            # Status Hero Bar
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                id="training-hub-status-text",
                                children=status_text,
                                style={
                                    "display": "inline-block",
                                    "padding": "6px 16px",
                                    "backgroundColor": status_color,
                                    "color": "#ffffff",
                                    "borderRadius": "20px",
                                    "fontSize": "13px",
                                    "fontWeight": "700",
                                    "letterSpacing": "0.5px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                        ],
                    ),
                    html.Div(
                        id="training-hub-status-metrics",
                        children="Ready to train",
                        style={
                            "display": "inline-block",
                            "marginLeft": "24px",
                            "fontSize": "14px",
                            "fontFamily": "ui-monospace, monospace",
                            "color": "#4A4A4A",
                        },
                    ),
                ],
                id="training-hub-status-hero",
                style={
                    "padding": "16px 32px",
                    "backgroundColor": "#f5f5f5",
                    "borderBottom": "1px solid #C6C6C6",
                },
            ),
            
            # Main Content Area (2-column: Config left 40%, Progress right 60%)
            html.Div(
                [
                    # Left Column (40% - Training Configuration & Controls)
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div("Training Configuration", style={
                                        "fontSize": "17px",
                                        "fontWeight": "700",
                                        "color": "#000000",
                                        "marginBottom": "20px",
                                        "fontFamily": "'Open Sans', Verdana, sans-serif",
                                    }),
                                    
                                    # Epochs input
                                    html.Div(
                                        [
                                            html.Label("Epochs", style={
                                                "fontSize": "14px",
                                                "color": "#6F6F6F",
                                                "display": "block",
                                                "marginBottom": "6px",
                                                "fontFamily": "'Open Sans', Verdana, sans-serif",
                                                "fontWeight": "600",
                                            }),
                                            dcc.Input(
                                                id="training-hub-epochs",
                                                type="number",
                                                min=1,
                                                max=200,
                                                step=1,
                                                value=target_epochs,
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
                                        style={"marginBottom": "16px"},
                                    ),
                                    
                                    # Essential Parameters (Collapsible)
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Span("Essential Parameters", style={
                                                        "fontSize": "14px",
                                                        "fontWeight": "600",
                                                        "color": "#6F6F6F",
                                                    }),
                                                    html.Span(id="training-hub-params-toggle", children="▼", style={
                                                        "marginLeft": "auto",
                                                        "fontSize": "12px",
                                                        "color": "#6F6F6F",
                                                    }),
                                                ],
                                                id="training-hub-params-header",
                                                n_clicks=0,
                                                style={
                                                    "display": "flex",
                                                    "alignItems": "center",
                                                    "padding": "8px 0",
                                                    "cursor": "pointer",
                                                    "userSelect": "none",
                                                    "marginBottom": "12px",
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    # Learning Rate
                                                    html.Div(
                                                        [
                                                            html.Label("Learning Rate", style={
                                                                "fontSize": "13px",
                                                                "color": "#6F6F6F",
                                                                "display": "block",
                                                                "marginBottom": "4px",
                                                                "fontWeight": "600",
                                                            }),
                                                            dcc.Input(
                                                                id="training-hub-lr",
                                                                type="number",
                                                                min=0.0001,
                                                                max=0.01,
                                                                step=0.0001,
                                                                value=float(np.clip(config.learning_rate, 0.0001, 0.01)),
                                                                debounce=True,
                                                                style={
                                                                    "width": "100%",
                                                                    "padding": "8px 10px",
                                                                    "fontSize": "13px",
                                                                    "border": "1px solid #C6C6C6",
                                                                    "borderRadius": "6px",
                                                                    "fontFamily": "ui-monospace, monospace",
                                                                },
                                                            ),
                                                        ],
                                                        style={"marginBottom": "12px"},
                                                    ),
                                                    
                                                    # Recon Weight
                                                    html.Div(
                                                        [
                                                            html.Label("Reconstruction Weight", style={
                                                                "fontSize": "13px",
                                                                "color": "#6F6F6F",
                                                                "display": "block",
                                                                "marginBottom": "4px",
                                                                "fontWeight": "600",
                                                            }),
                                                            dcc.Input(
                                                                id="training-hub-recon",
                                                                type="number",
                                                                min=0,
                                                                max=5000,
                                                                step=50,
                                                                value=float(np.clip(config.recon_weight, 0, 5000)),
                                                                debounce=True,
                                                                style={
                                                                    "width": "100%",
                                                                    "padding": "8px 10px",
                                                                    "fontSize": "13px",
                                                                    "border": "1px solid #C6C6C6",
                                                                    "borderRadius": "6px",
                                                                    "fontFamily": "ui-monospace, monospace",
                                                                },
                                                            ),
                                                        ],
                                                        style={"marginBottom": "12px"},
                                                    ),
                                                    
                                                    # KL Weight
                                                    html.Div(
                                                        [
                                                            html.Label("KL Weight", style={
                                                                "fontSize": "13px",
                                                                "color": "#6F6F6F",
                                                                "display": "block",
                                                                "marginBottom": "4px",
                                                                "fontWeight": "600",
                                                            }),
                                                            dcc.Input(
                                                                id="training-hub-kl",
                                                                type="number",
                                                                min=0.0,
                                                                max=1.0,
                                                                step=0.01,
                                                                value=float(np.clip(config.kl_weight, 0, 1)),
                                                                debounce=True,
                                                                style={
                                                                    "width": "100%",
                                                                    "padding": "8px 10px",
                                                                    "fontSize": "13px",
                                                                    "border": "1px solid #C6C6C6",
                                                                    "borderRadius": "6px",
                                                                    "fontFamily": "ui-monospace, monospace",
                                                                },
                                                            ),
                                                        ],
                                                    ),
                                                ],
                                                id="training-hub-params-content",
                                                style={"display": "block"},
                                            ),
                                        ],
                                        style={"marginBottom": "24px"},
                                    ),
                                    
                                    # Start Training Button
                                    dbc.Button(
                                        "Start Training",
                                        id="training-hub-start-button",
                                        n_clicks=0,
                                        style={
                                            "width": "100%",
                                            "height": "48px",
                                            "backgroundColor": "#C10A27",
                                            "border": "none",
                                            "borderRadius": "8px",
                                            "fontSize": "16px",
                                            "fontWeight": "700",
                                            "color": "#ffffff",
                                            "cursor": "pointer",
                                            "marginBottom": "12px",
                                        },
                                    ),
                                    
                                    # Stop Training Button
                                    dbc.Button(
                                        "Stop Training",
                                        id="training-hub-stop-button",
                                        n_clicks=0,
                                        disabled=True,
                                        style={
                                            "width": "100%",
                                            "height": "44px",
                                            "backgroundColor": "#ffffff",
                                            "border": "1px solid #C6C6C6",
                                            "borderRadius": "8px",
                                            "fontSize": "14px",
                                            "fontWeight": "600",
                                            "color": "#6F6F6F",
                                            "cursor": "pointer",
                                            "marginBottom": "24px",
                                        },
                                    ),
                                    
                                    # Link to Advanced Config
                                    dcc.Link(
                                        "Model Architecture & Advanced Settings →",
                                        href=f"/model/{model_id}/configure-training",
                                        style={
                                            "display": "block",
                                            "textAlign": "center",
                                            "fontSize": "13px",
                                            "color": "#45717A",
                                            "textDecoration": "none",
                                            "fontWeight": "600",
                                            "padding": "12px",
                                            "border": "1px solid #C6C6C6",
                                            "borderRadius": "6px",
                                            "backgroundColor": "#ffffff",
                                        },
                                    ),
                                ],
                                style={
                                    "padding": "24px",
                                    "backgroundColor": "#ffffff",
                                    "borderRadius": "8px",
                                },
                            ),
                        ],
                        style={
                            "width": "40%",
                            "paddingRight": "12px",
                        },
                    ),
                    
                    # Right Column (60% - Training Progress)
                    html.Div(
                        [
                            # Loss Curves
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span("Training Progress", style={
                                                "fontSize": "17px",
                                                "fontWeight": "700",
                                                "color": "#000000",
                                                "fontFamily": "'Open Sans', Verdana, sans-serif",
                                            }),
                                            dbc.Checkbox(
                                                id="training-hub-loss-smoothing",
                                                label="Smooth",
                                                value=[],
                                                style={
                                                    "marginLeft": "auto",
                                                    "fontSize": "13px",
                                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                                },
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                            "padding": "12px 24px",
                                            "borderBottom": "1px solid #C6C6C6",
                                            "backgroundColor": "#ffffff",
                                        },
                                    ),
                                    dcc.Graph(
                                        id="training-hub-loss-curves",
                                        style={"height": "400px", "width": "100%"},
                                        config={"displayModeBar": False},
                                    ),
                                ],
                                style={
                                    "backgroundColor": "#ffffff",
                                    "borderRadius": "8px",
                                    "overflow": "hidden",
                                },
                            ),
                        ],
                        style={
                            "width": "60%",
                            "paddingLeft": "12px",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "padding": "32px",
                    "gap": "24px",
                },
            ),
            
            # Training Terminal (Full-width)
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Training Terminal", style={
                                "fontSize": "15px",
                                "fontWeight": "700",
                                "color": "#000000",
                                "fontFamily": "'Open Sans', Verdana, sans-serif",
                            }),
                            html.Div(
                                [
                                    dbc.Button(
                                        "Clear",
                                        id="training-hub-terminal-clear",
                                        n_clicks=0,
                                        size="sm",
                                        style={
                                            "fontSize": "12px",
                                            "padding": "4px 12px",
                                            "backgroundColor": "#ffffff",
                                            "border": "1px solid #C6C6C6",
                                            "color": "#6F6F6F",
                                            "marginRight": "8px",
                                        },
                                    ),
                                    dbc.Button(
                                        "Download Logs",
                                        id="training-hub-terminal-download",
                                        n_clicks=0,
                                        size="sm",
                                        style={
                                            "fontSize": "12px",
                                            "padding": "4px 12px",
                                            "backgroundColor": "#ffffff",
                                            "border": "1px solid #C6C6C6",
                                            "color": "#6F6F6F",
                                        },
                                    ),
                                ],
                                style={"marginLeft": "auto", "display": "flex"},
                            ),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "padding": "12px 24px",
                            "borderBottom": "2px solid #C10A27",
                            "backgroundColor": "#ffffff",
                        },
                    ),
                    html.Div(
                        id="training-hub-terminal",
                        children=[
                            html.Div("Ready to train. Click 'Start Training' to begin.", style={
                                "fontFamily": "ui-monospace, 'SF Mono', Monaco, monospace",
                                "fontSize": "13px",
                                "color": "#d4d4d4",
                                "lineHeight": "1.6",
                            }),
                        ],
                        style={
                            "padding": "16px 24px",
                            "backgroundColor": "#1e1e1e",
                            "color": "#d4d4d4",
                            "fontFamily": "ui-monospace, 'SF Mono', Monaco, monospace",
                            "fontSize": "13px",
                            "minHeight": "300px",
                            "maxHeight": "400px",
                            "overflowY": "auto",
                            "lineHeight": "1.6",
                        },
                    ),
                ],
                style={
                    "margin": "0 32px 32px 32px",
                    "backgroundColor": "#ffffff",
                    "borderRadius": "8px",
                    "overflow": "hidden",
                    "border": "1px solid #C6C6C6",
                },
            ),
        ],
        style={
            "fontFamily": "'Open Sans', Verdana, sans-serif",
            "backgroundColor": "#fafafa",
            "minHeight": "100vh",
        },
    )
