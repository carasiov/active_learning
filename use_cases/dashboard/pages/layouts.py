from __future__ import annotations

from dash import dcc, html
import dash_bootstrap_components as dbc
import numpy as np

from use_cases.dashboard.core import state as dashboard_state
from use_cases.dashboard.core.model_manager import ModelManager


def build_dashboard_layout() -> html.Div:
    """Build dashboard with intelligent proportional panel resizing."""
    dashboard_state.initialize_model_and_data()
    
    with dashboard_state.state_manager.state_lock:
        # Check if we have an active model
        if dashboard_state.state_manager.state.active_model is None:
            # Return a simple message if no model loaded
            return html.Div([
                html.H3("No Model Loaded", style={"textAlign": "center", "marginTop": "100px"}),
                html.P("Please select a model from the home page.", style={"textAlign": "center"}),
                html.A("Go to Home", href="/", style={"display": "block", "textAlign": "center"})
            ])
        
        config = dashboard_state.state_manager.state.active_model.config
        default_epochs = max(1, dashboard_state.state_manager.state.active_model.training.target_epochs or 5)
        latent_version = dashboard_state.state_manager.state.active_model.data.version
        existing_status = list(dashboard_state.state_manager.state.active_model.training.status_messages)
        selected_sample = dashboard_state.state_manager.state.active_model.ui.selected_sample
        labels_version = dashboard_state.state_manager.state.active_model.data.version
        model_id = dashboard_state.state_manager.state.active_model.model_id

    status_initial = existing_status[-3:] if existing_status else ["Ready to train"]
    checkpoint_path = ModelManager.checkpoint_path(model_id)
    try:
        checkpoint_display = str(checkpoint_path.relative_to(dashboard_state.ROOT_DIR))
    except ValueError:
        checkpoint_display = str(checkpoint_path)

    return html.Div(
        [
            # Hidden stores and intervals - use session storage to persist across navigation
            dcc.Store(id="selected-sample-store", data=selected_sample, storage_type='session'),
            dcc.Store(id="labels-store", data={"version": labels_version}, storage_type='session'),
            dcc.Store(id="training-control-store", data={"token": 0}, storage_type='session'),
            dcc.Store(id="latent-store", data={"version": latent_version}, storage_type='session'),
            dcc.Interval(id="training-poll", interval=2000, n_intervals=0, disabled=True),
            dcc.Store(id="keyboard-label-store"),
            dcc.Interval(id="keyboard-poll", interval=300, n_intervals=0, disabled=False),
            dcc.Interval(id='resize-setup-trigger', interval=100, n_intervals=0, max_intervals=1),
            
            # Training confirmation modal
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Confirm Training")),
                    dbc.ModalBody(
                        [
                            html.Div(id="modal-training-info", style={
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
                                id="modal-cancel-button",
                                n_clicks=0,
                                style={
                                    "backgroundColor": "#ffffff",
                                    "color": "#6F6F6F",
                                    "border": "1px solid #C6C6C6",
                                    "borderRadius": "6px",
                                    "padding": "8px 16px",
                                    "fontSize": "13px",
                                    "fontWeight": "600",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                            dbc.Button(
                                "Start Training",
                                id="modal-confirm-button",
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
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                        ]
                    ),
                ],
                id="training-confirm-modal",
                is_open=False,
                centered=True,
            ),
            
            # Header with logo and red accent bar
            html.Div([
                # Top header with logo
                html.Div(
                    [
                        # infoteam Logo (left)
                        html.Div(
                            [
                                dcc.Link(
                                    html.Img(
                                        src="/assets/infoteam_logo_basic.png",
                                        alt="infoteam software",
                                        style={
                                            "height": "50px",
                                            "width": "auto",
                                            "display": "block",
                                        },
                                    ),
                                    href="/",
                                    style={"textDecoration": "none", "display": "block"},
                                ),
                            ],
                            style={"display": "inline-block", "marginRight": "32px"},
                        ),
                        # Title (right)
                        html.Div(
                            [
                                html.H1("SSVAE Active Learning", style={
                                    "fontSize": "24px",
                                    "fontWeight": "700",
                                    "margin": "0",
                                    "color": "#000000",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                }),
                                html.Div("MNIST Semi-Supervised Learning", style={
                                    "fontSize": "15px",
                                    "color": "#6F6F6F",
                                    "marginTop": "2px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                }),
                            ],
                            style={
                                "display": "inline-block",
                                "marginLeft": "0",
                                "verticalAlign": "middle",
                            },
                        ),
                    ],
                    style={
                        "padding": "16px 32px",
                        "backgroundColor": "#ffffff",
                        "display": "flex",
                        "alignItems": "center",
                    },
                ),
                # Red accent bar (infoteam brand element)
                html.Div(style={
                    "height": "4px",
                    "backgroundColor": "#C10A27",
                    "width": "100%",
                }),
            ]),
            
            # Main resizable layout
            html.Div(
                [
                    # LEFT PANEL
                    html.Div(
                        [
                            html.Div(
                                [
                                    _build_stats_section(),
                                    _build_run_history_section(model_id),
                                    _build_status_section(status_initial),

                                    # Training section
                                    html.Div(
                                        [
                                            html.Div("Training", style={
                                                "fontSize": "15px",
                                                "fontWeight": "700",
                                                "color": "#000000",
                                                "marginBottom": "16px",
                                                "fontFamily": "'Open Sans', Verdana, sans-serif",
                                            }),
                                            
                                            # Epochs input
                                            html.Div(
                                                [
                                                    html.Label("Epochs", style={
                                                        "fontSize": "13px",
                                                        "color": "#6F6F6F",
                                                        "display": "block",
                                                        "marginBottom": "6px",
                                                        "fontFamily": "'Open Sans', Verdana, sans-serif",
                                                        "fontWeight": "600",
                                                    }),
                                                    dcc.Input(
                                                        id="num-epochs-input",
                                                        type="number",
                                                        min=1,
                                                        max=200,
                                                        step=1,
                                                        value=default_epochs,
                                                        placeholder="e.g., 10",
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
                                                style={"marginBottom": "16px"},
                                            ),
                                            
                                            # Train button
                                            dbc.Button(
                                                "Train Model",
                                                id="start-training-button",
                                                n_clicks=0,
                                                style={
                                                    "width": "100%",
                                                    "height": "44px",
                                                    "backgroundColor": "#C10A27",
                                                    "border": "none",
                                                    "borderRadius": "6px",
                                                    "fontSize": "14px",
                                                    "fontWeight": "700",
                                                    "color": "#ffffff",
                                                    "cursor": "pointer",
                                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                                    "marginBottom": "12px",
                                                },
                                            ),
                                            
                                            # Link to Training Hub
                                            dcc.Link(
                                                "Training Hub →",
                                                href=f"/model/{model_id}/training-hub",
                                                style={
                                                    "display": "block",
                                                    "textAlign": "center",
                                                    "fontSize": "13px",
                                                    "color": "#45717A",
                                                    "textDecoration": "none",
                                                    "fontWeight": "600",
                                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                                    "padding": "8px",
                                                },
                                            ),
                                        ],
                                        style={
                                            "marginTop": "auto",
                                            "paddingTop": "16px",
                                            "borderTop": "1px solid #C6C6C6",
                                        },
                                    ),
                                ],
                                style={
                                    "padding": "24px",
                                    "height": "100%",
                                    "display": "flex",
                                    "flexDirection": "column",
                                    "overflowY": "auto",
                                },
                            ),
                        ],
                        id="left-panel",
                        style={
                            "width": "20%",
                            "minWidth": "240px",
                            "backgroundColor": "#ffffff",
                            "borderRight": "1px solid #C6C6C6",
                        },
                    ),
                    
                    # LEFT RESIZE HANDLE
                    html.Div(
                        id="left-resize-handle",
                        className="resize-handle",
                        style={
                            "width": "5px",
                            "cursor": "col-resize",
                            "backgroundColor": "transparent",
                            "flexShrink": "0",
                        },
                    ),
                    
                    # CENTER PANEL
                    html.Div(
                        [
                            # Color mode selector + legend
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span("Color by:", style={
                                                "fontSize": "14px",
                                                "color": "#6F6F6F",
                                                "marginRight": "12px",
                                                "fontFamily": "'Open Sans', Verdana, sans-serif",
                                            }),
                                            dbc.RadioItems(
                                                id="color-mode-radio",
                                                options=[
                                                    {"label": "Labels", "value": "user_labels"},
                                                    {"label": "Predicted", "value": "pred_class"},
                                                    {"label": "True", "value": "true_class"},
                                                    {"label": "Confidence", "value": "certainty"},
                                                ],
                                                value="user_labels",
                                                inline=True,
                                                style={"fontSize": "14px", "fontFamily": "'Open Sans', Verdana, sans-serif"},
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                        },
                                    ),
                                    html.Div(
                                        id="scatter-legend",
                                        style={
                                            "marginTop": "8px",
                                        },
                                    ),
                                ],
                                style={
                                    "padding": "12px 24px",
                                    "borderBottom": "1px solid #C6C6C6",
                                    "backgroundColor": "#ffffff",
                                    "flexShrink": "0",
                                },
                            ),
                            
                            # Latent space
                            html.Div(
                                [
                                    html.Div(
                                        "2D Latent Space (SSVAE Encoder)",
                                        style={
                                            "position": "absolute",
                                            "top": "8px",
                                            "left": "24px",
                                            "fontSize": "16px",
                                            "fontWeight": "700",
                                            "color": "#000000",
                                            "backgroundColor": "rgba(255, 255, 255, 0.9)",
                                            "padding": "4px 8px",
                                            "borderRadius": "4px",
                                            "zIndex": "1000",
                                            "fontFamily": "'Open Sans', Verdana, sans-serif",
                                        },
                                    ),
                                    dcc.Graph(
                                        id="latent-scatter",
                                        style={"height": "100%", "width": "100%"},
                                        config={
                                            "displayModeBar": True,
                                            "scrollZoom": True,
                                            "responsive": True,
                                            "displaylogo": False,
                                        },
                                    ),
                                ],
                                id="latent-plot-container",
                                style={
                                    "flex": "1",
                                    "minHeight": "400px",
                                    "backgroundColor": "#fafafa",
                                    "position": "relative",
                                },
                            ),
                        ],
                        id="center-panel",
                        style={
                            "flex": "1",
                            "minWidth": "500px",
                            "display": "flex",
                            "flexDirection": "column",
                        },
                    ),
                    
                    # RIGHT RESIZE HANDLE
                    html.Div(
                        id="right-resize-handle",
                        className="resize-handle",
                        style={
                            "width": "5px",
                            "cursor": "col-resize",
                            "backgroundColor": "transparent",
                            "flexShrink": "0",
                        },
                    ),
                    
                    # RIGHT PANEL
                    html.Div(
                        [
                            html.Div(
                                id="selected-sample-header",
                                children="Select a point",
                                style={
                                    "fontSize": "17px",
                                    "fontWeight": "700",
                                    "color": "#000000",
                                    "padding": "16px 24px",
                                    "borderBottom": "2px solid #C10A27",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                            
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div("Original", style={
                                                "fontSize": "13px",
                                                "color": "#6F6F6F",
                                                "marginBottom": "8px",
                                                "textAlign": "center",
                                                "textTransform": "uppercase",
                                                "letterSpacing": "0.5px",
                                                "fontFamily": "'Open Sans', Verdana, sans-serif",
                                                "fontWeight": "600",
                                            }),
                                            html.Div(
                                                html.Img(
                                                    id="original-image",
                                                    style={
                                                        "width": "112px",
                                                        "height": "112px",
                                                        "imageRendering": "pixelated",
                                                    },
                                                ),
                                                style={"textAlign": "center"},
                                            ),
                                        ],
                                        style={"flex": "1"},
                                    ),
                                    html.Div(
                                        [
                                            html.Div("Reconstructed", style={
                                                "fontSize": "13px",
                                                "color": "#6F6F6F",
                                                "marginBottom": "8px",
                                                "textAlign": "center",
                                                "textTransform": "uppercase",
                                                "letterSpacing": "0.5px",
                                                "fontFamily": "'Open Sans', Verdana, sans-serif",
                                                "fontWeight": "600",
                                            }),
                                            html.Div(
                                                html.Img(
                                                    id="reconstructed-image",
                                                    style={
                                                        "width": "112px",
                                                        "height": "112px",
                                                        "imageRendering": "pixelated",
                                                    },
                                                ),
                                                style={"textAlign": "center"},
                                            ),
                                        ],
                                        style={"flex": "1"},
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "gap": "16px",
                                    "padding": "20px 24px",
                                },
                            ),
                            
                            html.Div(
                                id="prediction-info",
                                style={
                                    "padding": "12px 24px",
                                    "backgroundColor": "#f5f5f5",
                                    "fontSize": "13px",
                                    "fontFamily": "ui-monospace, 'SF Mono', Monaco, monospace",
                                    "color": "#4A4A4A",
                                    "lineHeight": "1.8",
                                    "borderTop": "1px solid #C6C6C6",
                                    "borderBottom": "1px solid #C6C6C6",
                                },
                            ),
                            
                            html.Div(
                                [
                                    html.Div("Assign Label", style={
                                        "fontSize": "15px",
                                        "fontWeight": "700",
                                        "color": "#000000",
                                        "marginBottom": "12px",
                                        "fontFamily": "'Open Sans', Verdana, sans-serif",
                                    }),
                                    
                                    # Single row of digit buttons with flex wrap
                                    html.Div(
                                        [
                                            dbc.Button(
                                                str(d),
                                                id={"type": "label-button", "label": d},
                                                n_clicks=0,
                                                style={
                                                    "width": "44px",
                                                    "height": "44px",
                                                    "padding": "0",
                                                    "fontSize": "16px",
                                                    "fontWeight": "700",
                                                    "backgroundColor": "#ffffff",
                                                    "color": "#C10A27",
                                                    "border": "1.5px solid #C10A27",
                                                    "borderRadius": "6px",
                                                    "cursor": "pointer",
                                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                                    "flexShrink": "0",
                                                },
                                            )
                                            for d in range(10)
                                        ],
                                        style={
                                            "display": "flex",
                                            "flexWrap": "wrap",
                                            "gap": "8px",
                                        },
                                    ),
                                    
                                    dbc.Button(
                                        "Clear Label",
                                        id="delete-label-button",
                                        n_clicks=0,
                                        style={
                                            "width": "100%",
                                            "height": "36px",
                                            "marginTop": "12px",
                                            "fontSize": "14px",
                                            "fontWeight": "600",
                                            "backgroundColor": "#ffffff",
                                            "color": "#6F6F6F",
                                            "border": "1px solid #C6C6C6",
                                            "borderRadius": "6px",
                                            "cursor": "pointer",
                                            "fontFamily": "'Open Sans', Verdana, sans-serif",
                                        },
                                    ),
                                    
                                    html.Div(
                                        id="label-feedback",
                                        style={
                                            "marginTop": "12px",
                                            "fontSize": "13px",
                                            "color": "#6F6F6F",
                                            "textAlign": "center",
                                            "minHeight": "18px",
                                        },
                                    ),
                                ],
                                style={"padding": "20px 24px"},
                            ),
                            
                            html.Div(
                                [
                                    html.Div("Keyboard Shortcuts", style={
                                        "fontSize": "13px",
                                        "fontWeight": "700",
                                        "color": "#6F6F6F",
                                        "marginBottom": "8px",
                                        "textTransform": "uppercase",
                                        "letterSpacing": "0.5px",
                                        "fontFamily": "'Open Sans', Verdana, sans-serif",
                                    }),
                                    html.Div("0–9: Assign label", style={
                                        "fontSize": "13px",
                                        "color": "#4A4A4A",
                                        "lineHeight": "1.6",
                                        "fontFamily": "'Open Sans', Verdana, sans-serif",
                                    }),
                                    html.Div("Tab: Navigate controls", style={
                                        "fontSize": "13px",
                                        "color": "#4A4A4A",
                                        "lineHeight": "1.6",
                                        "fontFamily": "'Open Sans', Verdana, sans-serif",
                                    }),
                                ],
                                style={
                                    "marginTop": "auto",
                                    "padding": "16px 24px",
                                    "borderTop": "1px solid #C6C6C6",
                                    "backgroundColor": "#f5f5f5",
                                },
                            ),
                        ],
                        id="right-panel",
                        style={
                            "width": "20%",
                            "minWidth": "240px",
                            "backgroundColor": "#ffffff",
                            "borderLeft": "1px solid #C6C6C6",
                            "display": "flex",
                            "flexDirection": "column",
                            "overflowY": "auto",
                        },
                    ),
                ],
                id="main-container",
                style={
                    "display": "flex",
                    "flex": "1",
                    "overflow": "hidden",
                    "minHeight": "0",
                },
            ),
        ],
        style={
            "fontFamily": "'Open Sans', Verdana, sans-serif",
            "backgroundColor": "#fafafa",
            "minWidth": "1200px",
            "height": "100vh",
            "overflow": "hidden",
            "display": "flex",
            "flexDirection": "column",
        },
    )


def _build_stats_section() -> html.Div:
    return html.Div(
        [
            html.Div("Dataset", style={
                "fontSize": "15px",
                "fontWeight": "700",
                "color": "#000000",
                "marginBottom": "12px",
                "fontFamily": "'Open Sans', Verdana, sans-serif",
            }),
            html.Div(
                id="dataset-stats",
                style={"fontSize": "13px", "lineHeight": "1.6", "fontFamily": "'Open Sans', Verdana, sans-serif"},
            ),
        ],
        style={
            "marginBottom": "24px",
            "paddingBottom": "24px",
            "borderBottom": "1px solid #C6C6C6",
        },
    )


def _run_history_placeholder() -> html.Div:
    return html.Div(
        "No completed training runs yet.",
        style={
            "fontSize": "12px",
            "color": "#6F6F6F",
            "fontStyle": "italic",
        },
    )


def _build_run_history_section(model_id: str) -> html.Div:
    return html.Div(
        [
            html.Div(
                "Training Runs (latest 8)",
                style={
                    "fontSize": "15px",
                    "fontWeight": "700",
                    "color": "#000000",
                    "marginBottom": "12px",
                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                },
            ),
            html.Div(
                [_run_history_placeholder()],
                id="run-history-list",
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "gap": "10px",
                },
            ),
            dcc.Link(
                "Open full history →",
                href=f"/experiments?model={model_id}",
                style={
                    "fontSize": "12px",
                    "color": "#45717A",
                    "textDecoration": "none",
                    "fontWeight": "600",
                    "marginTop": "12px",
                    "display": "inline-block",
                },
            ),
        ],
        style={
            "marginBottom": "24px",
            "paddingBottom": "24px",
            "borderBottom": "1px solid #C6C6C6",
        },
    )


def _build_config_section(config, default_epochs: int, model_id: str | None = None) -> html.Div:
    advanced_config_href = (
        f"/model/{model_id}/configure-training" if model_id else "/configure-training"
    )
    return html.Div(
        [
            # Link to advanced configuration page
            html.Div(
                dcc.Link(
                    "⚙️ Advanced Configuration",
                    href=advanced_config_href,
                    style={
                        "fontSize": "13px",
                        "color": "#C10A27",
                        "textDecoration": "none",
                        "fontWeight": "600",
                        "display": "block",
                        "marginBottom": "16px",
                        "fontFamily": "'Open Sans', Verdana, sans-serif",
                    },
                ),
            ),
            
            html.Div("Training Configuration", style={
                "fontSize": "15px",
                "fontWeight": "700",
                "color": "#000000",
                "marginBottom": "12px",
                "fontFamily": "'Open Sans', Verdana, sans-serif",
            }),
            
            html.Div(
                [
                    html.Label("Epochs", style={
                        "fontSize": "13px",
                        "color": "#6F6F6F",
                        "display": "block",
                        "marginBottom": "4px",
                        "fontFamily": "'Open Sans', Verdana, sans-serif",
                        "fontWeight": "600",
                    }),
                    dcc.Input(
                        id="num-epochs-input",
                        type="number",
                        min=1,
                        max=200,
                        step=1,
                        value=default_epochs,
                        placeholder="e.g., 5",
                        debounce=True,
                        style={
                            "width": "100%",
                            "padding": "6px 8px",
                            "fontSize": "13px",
                            "border": "1px solid #C6C6C6",
                            "borderRadius": "6px",
                            "fontFamily": "ui-monospace, monospace",
                        },
                    ),
                ],
                style={"marginBottom": "12px"},
            ),
            
            html.Div(
                [
                    html.Label("Learning Rate", style={
                        "fontSize": "13px",
                        "color": "#6F6F6F",
                        "display": "block",
                        "marginBottom": "4px",
                        "fontFamily": "'Open Sans', Verdana, sans-serif",
                        "fontWeight": "600",
                    }),
                    dcc.Input(
                        id="learning-rate-slider",
                        type="number",
                        min=0.0001,
                        max=0.01,
                        step=0.0001,
                        value=float(np.clip(config.learning_rate, 0.0001, 0.01)),
                        debounce=True,
                        style={
                            "width": "100%",
                            "padding": "6px 8px",
                            "fontSize": "13px",
                            "border": "1px solid #C6C6C6",
                            "borderRadius": "6px",
                            "fontFamily": "ui-monospace, monospace",
                        },
                    ),
                ],
                style={"marginBottom": "12px"},
            ),
            
            html.Div(
                [
                    html.Label([
                        "Recon Weight",
                        html.Span(" (higher = better image quality)", style={
                            "fontSize": "11px",
                            "color": "#6F6F6F",
                            "fontWeight": "normal",
                        }),
                    ], style={
                        "fontSize": "13px",
                        "color": "#6F6F6F",
                        "display": "block",
                        "marginBottom": "4px",
                        "fontFamily": "'Open Sans', Verdana, sans-serif",
                        "fontWeight": "600",
                    }),
                    dcc.Input(
                        id="recon-weight-slider",
                        type="number",
                        min=0,
                        max=5000,
                        step=50,
                        value=float(np.clip(config.recon_weight, 0.0, 5000.0)),
                        debounce=True,
                        style={
                            "width": "100%",
                            "padding": "6px 8px",
                            "fontSize": "13px",
                            "border": "1px solid #C6C6C6",
                            "borderRadius": "6px",
                            "fontFamily": "ui-monospace, monospace",
                        },
                    ),
                ],
                style={"marginBottom": "12px"},
            ),
            
            html.Div(
                [
                    html.Label("KL Weight", style={
                        "fontSize": "13px",
                        "color": "#6F6F6F",
                        "display": "block",
                        "marginBottom": "4px",
                        "fontFamily": "'Open Sans', Verdana, sans-serif",
                        "fontWeight": "600",
                    }),
                    dcc.Input(
                        id="kl-weight-slider",
                        type="number",
                        min=0.0,
                        max=1.0,
                        step=0.01,
                        value=float(np.clip(config.kl_weight, 0.0, 1.0)),
                        debounce=True,
                        style={
                            "width": "100%",
                            "padding": "6px 8px",
                            "fontSize": "13px",
                            "border": "1px solid #C6C6C6",
                            "borderRadius": "6px",
                            "fontFamily": "ui-monospace, monospace",
                        },
                    ),
                ],
            ),
        ],
        style={
            "marginBottom": "24px",
            "paddingBottom": "24px",
            "borderBottom": "1px solid #C6C6C6",
        },
    )


def _build_status_section(status_messages: list) -> html.Div:
    return html.Div(
        [
            html.Div("Status", style={
                "fontSize": "15px",
                "fontWeight": "700",
                "color": "#000000",
                "marginBottom": "8px",
                "fontFamily": "'Open Sans', Verdana, sans-serif",
            }),
            html.Div(
                id="training-status",
                children=[
                    html.Div(msg, style={
                        "fontSize": "12px",
                        "color": "#6F6F6F",
                        "fontFamily": "ui-monospace, monospace",
                        "lineHeight": "1.6",
                    })
                    for msg in status_messages
                ],
                style={
                    "padding": "8px",
                    "backgroundColor": "#f5f5f5",
                    "borderRadius": "6px",
                    "minHeight": "80px",
                    "maxHeight": "180px",
                    "overflowY": "auto",
                },
            ),
        ],
        style={"marginBottom": "24px"},
    )
