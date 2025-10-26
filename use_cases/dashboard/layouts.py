from __future__ import annotations

from dash import dcc, html
import dash_bootstrap_components as dbc
import numpy as np

from use_cases.dashboard.state import app_state, state_lock, initialize_model_and_data


def build_dashboard_layout() -> html.Div:
    """Build dashboard with intelligent proportional panel resizing."""
    initialize_model_and_data()
    
    with state_lock:
        config = app_state["config"]
        default_epochs = max(1, int(app_state["training"]["target_epochs"] or 5))
        latent_version = int(app_state["ui"]["latent_version"])
        existing_status = list(app_state["training"]["status_messages"])

    status_initial = existing_status[-3:] if existing_status else ["Ready to train"]

    return html.Div(
        [
            # Hidden stores and intervals
            dcc.Store(id="selected-sample-store", data=int(app_state["ui"]["selected_sample"])),
            dcc.Store(id="labels-store", data={"version": int(app_state["ui"]["labels_version"])}),
            dcc.Store(id="training-control-store", data={"token": 0}),
            dcc.Store(id="latent-store", data={"version": latent_version}),
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
                                "fontSize": "14px",
                                "lineHeight": "1.6",
                                "color": "#1d1d1f",
                            }),
                            html.Div(
                                "⚠️ This will overwrite the current checkpoint at ssvae.ckpt",
                                style={
                                    "marginTop": "16px",
                                    "padding": "12px",
                                    "backgroundColor": "#FFF3CD",
                                    "border": "1px solid #FFE69C",
                                    "borderRadius": "6px",
                                    "fontSize": "13px",
                                    "color": "#856404",
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
                                    "color": "#86868b",
                                    "border": "1px solid #d1d1d6",
                                    "borderRadius": "6px",
                                    "padding": "8px 16px",
                                    "fontSize": "13px",
                                    "fontWeight": "500",
                                },
                            ),
                            dbc.Button(
                                "Start Training",
                                id="modal-confirm-button",
                                n_clicks=0,
                                style={
                                    "backgroundColor": "#34C759",
                                    "color": "#ffffff",
                                    "border": "none",
                                    "borderRadius": "6px",
                                    "padding": "8px 16px",
                                    "fontSize": "13px",
                                    "fontWeight": "600",
                                    "marginLeft": "8px",
                                },
                            ),
                        ]
                    ),
                ],
                id="training-confirm-modal",
                is_open=False,
                centered=True,
            ),
            
            # Header
            html.Div(
                [
                    html.H1("SSVAE Active Learning", style={
                        "fontSize": "20px",
                        "fontWeight": "600",
                        "margin": "0",
                        "color": "#1d1d1f",
                    }),
                    html.Div("MNIST Semi-Supervised Learning", style={
                        "fontSize": "13px",
                        "color": "#86868b",
                        "marginTop": "2px",
                    }),
                ],
                style={
                    "padding": "20px 32px",
                    "borderBottom": "1px solid #e5e5e5",
                    "backgroundColor": "#ffffff",
                },
            ),
            
            # Main resizable layout
            html.Div(
                [
                    # LEFT PANEL
                    html.Div(
                        [
                            html.Div(
                                [
                                    _build_stats_section(),
                                    _build_config_section(config, default_epochs),
                                    _build_status_section(status_initial),
                                    
                                    html.Div(
                                        dbc.Button(
                                            "Train Model",
                                            id="start-training-button",
                                            n_clicks=0,
                                            style={
                                                "width": "100%",
                                                "height": "44px",
                                                "backgroundColor": "#34C759",
                                                "border": "none",
                                                "borderRadius": "8px",
                                                "fontSize": "15px",
                                                "fontWeight": "600",
                                                "color": "#ffffff",
                                                "cursor": "pointer",
                                            },
                                        ),
                                        style={"marginTop": "auto", "paddingTop": "16px"},
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
                            "borderRight": "1px solid #e5e5e5",
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
                                                "fontSize": "12px",
                                                "color": "#86868b",
                                                "marginRight": "12px",
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
                                                style={"fontSize": "13px"},
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
                                    "borderBottom": "1px solid #e5e5e5",
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
                                            "fontSize": "12px",
                                            "fontWeight": "600",
                                            "color": "#1d1d1f",
                                            "backgroundColor": "rgba(255, 255, 255, 0.9)",
                                            "padding": "4px 8px",
                                            "borderRadius": "4px",
                                            "zIndex": "1000",
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
                                    "minHeight": "300px",
                                    "backgroundColor": "#fafafa",
                                    "position": "relative",
                                },
                            ),
                            
                            # HORIZONTAL RESIZE HANDLE
                            html.Div(
                                id="horizontal-resize-handle",
                                className="resize-handle",
                                style={
                                    "height": "5px",
                                    "cursor": "row-resize",
                                    "backgroundColor": "transparent",
                                    "flexShrink": "0",
                                },
                            ),
                            
                            # Loss curves with smoothing toggle
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span("Training Progress", style={
                                                "fontSize": "13px",
                                                "fontWeight": "600",
                                                "color": "#1d1d1f",
                                            }),
                                            dbc.Checkbox(
                                                id="loss-smoothing-toggle",
                                                label="Smooth",
                                                value=[],
                                                style={
                                                    "marginLeft": "auto",
                                                    "fontSize": "12px",
                                                },
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                            "padding": "8px 16px",
                                            "borderBottom": "1px solid #e5e5e5",
                                        },
                                    ),
                                    dcc.Graph(
                                        id="loss-curves",
                                        style={"height": "calc(100% - 45px)", "width": "100%"},
                                        config={"displayModeBar": False},
                                    ),
                                ],
                                id="loss-plot-container",
                                style={
                                    "height": "220px",
                                    "minHeight": "200px",
                                    "borderTop": "1px solid #e5e5e5",
                                    "backgroundColor": "#ffffff",
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
                                    "fontSize": "15px",
                                    "fontWeight": "600",
                                    "color": "#1d1d1f",
                                    "padding": "16px 24px",
                                    "borderBottom": "1px solid #e5e5e5",
                                },
                            ),
                            
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div("Original", style={
                                                "fontSize": "11px",
                                                "color": "#86868b",
                                                "marginBottom": "8px",
                                                "textAlign": "center",
                                                "textTransform": "uppercase",
                                                "letterSpacing": "0.5px",
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
                                                "fontSize": "11px",
                                                "color": "#86868b",
                                                "marginBottom": "8px",
                                                "textAlign": "center",
                                                "textTransform": "uppercase",
                                                "letterSpacing": "0.5px",
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
                                    "backgroundColor": "#f5f5f7",
                                    "fontSize": "12px",
                                    "fontFamily": "ui-monospace, 'SF Mono', Monaco, monospace",
                                    "color": "#1d1d1f",
                                    "lineHeight": "1.8",
                                    "borderTop": "1px solid #e5e5e5",
                                    "borderBottom": "1px solid #e5e5e5",
                                },
                            ),
                            
                            html.Div(
                                [
                                    html.Div("Assign Label", style={
                                        "fontSize": "13px",
                                        "fontWeight": "600",
                                        "color": "#1d1d1f",
                                        "marginBottom": "12px",
                                    }),
                                    
                                    html.Div(
                                        [
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
                                                            "fontWeight": "600",
                                                            "backgroundColor": "#ffffff",
                                                            "color": "#007AFF",
                                                            "border": "1.5px solid #007AFF",
                                                            "borderRadius": "8px",
                                                            "cursor": "pointer",
                                                        },
                                                    )
                                                    for d in range(5)
                                                ],
                                                style={
                                                    "display": "flex",
                                                    "gap": "8px",
                                                    "marginBottom": "8px",
                                                },
                                            ),
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
                                                            "fontWeight": "600",
                                                            "backgroundColor": "#ffffff",
                                                            "color": "#007AFF",
                                                            "border": "1.5px solid #007AFF",
                                                            "borderRadius": "8px",
                                                            "cursor": "pointer",
                                                        },
                                                    )
                                                    for d in range(5, 10)
                                                ],
                                                style={
                                                    "display": "flex",
                                                    "gap": "8px",
                                                },
                                            ),
                                        ],
                                    ),
                                    
                                    dbc.Button(
                                        "Clear Label",
                                        id="delete-label-button",
                                        n_clicks=0,
                                        style={
                                            "width": "100%",
                                            "height": "36px",
                                            "marginTop": "12px",
                                            "fontSize": "13px",
                                            "fontWeight": "500",
                                            "backgroundColor": "#ffffff",
                                            "color": "#FF3B30",
                                            "border": "1px solid #FF3B30",
                                            "borderRadius": "6px",
                                            "cursor": "pointer",
                                        },
                                    ),
                                    
                                    html.Div(
                                        id="label-feedback",
                                        style={
                                            "marginTop": "12px",
                                            "fontSize": "12px",
                                            "color": "#86868b",
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
                                        "fontSize": "11px",
                                        "fontWeight": "600",
                                        "color": "#86868b",
                                        "marginBottom": "8px",
                                        "textTransform": "uppercase",
                                        "letterSpacing": "0.5px",
                                    }),
                                    html.Div("0–9: Assign label", style={
                                        "fontSize": "12px",
                                        "color": "#1d1d1f",
                                        "lineHeight": "1.6",
                                    }),
                                    html.Div("Tab: Navigate controls", style={
                                        "fontSize": "12px",
                                        "color": "#1d1d1f",
                                        "lineHeight": "1.6",
                                    }),
                                ],
                                style={
                                    "marginTop": "auto",
                                    "padding": "16px 24px",
                                    "borderTop": "1px solid #e5e5e5",
                                    "backgroundColor": "#f5f5f7",
                                },
                            ),
                        ],
                        id="right-panel",
                        style={
                            "width": "20%",
                            "minWidth": "240px",
                            "backgroundColor": "#ffffff",
                            "borderLeft": "1px solid #e5e5e5",
                            "display": "flex",
                            "flexDirection": "column",
                            "overflowY": "auto",
                        },
                    ),
                ],
                id="main-container",
                style={
                    "display": "flex",
                    "height": "calc(100vh - 85px)",
                    "overflow": "hidden",
                },
            ),
        ],
        style={
            "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            "backgroundColor": "#fafafa",
            "minWidth": "1200px",
            "height": "100vh",
            "overflow": "hidden",
        },
    )


def _build_stats_section() -> html.Div:
    return html.Div(
        [
            html.Div("Dataset", style={
                "fontSize": "13px",
                "fontWeight": "600",
                "color": "#1d1d1f",
                "marginBottom": "12px",
            }),
            html.Div(
                id="dataset-stats",
                style={"fontSize": "12px", "lineHeight": "1.6"},
            ),
        ],
        style={
            "marginBottom": "24px",
            "paddingBottom": "24px",
            "borderBottom": "1px solid #e5e5e5",
        },
    )


def _build_config_section(config, default_epochs: int) -> html.Div:
    return html.Div(
        [
            html.Div("Training Configuration", style={
                "fontSize": "13px",
                "fontWeight": "600",
                "color": "#1d1d1f",
                "marginBottom": "12px",
            }),
            
            html.Div(
                [
                    html.Label("Epochs", style={
                        "fontSize": "12px",
                        "color": "#86868b",
                        "display": "block",
                        "marginBottom": "4px",
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
                            "border": "1px solid #d1d1d6",
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
                        "fontSize": "12px",
                        "color": "#86868b",
                        "display": "block",
                        "marginBottom": "4px",
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
                            "border": "1px solid #d1d1d6",
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
                            "fontSize": "10px",
                            "color": "#86868b",
                            "fontWeight": "normal",
                        }),
                    ], style={
                        "fontSize": "12px",
                        "color": "#86868b",
                        "display": "block",
                        "marginBottom": "4px",
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
                            "border": "1px solid #d1d1d6",
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
                        "fontSize": "12px",
                        "color": "#86868b",
                        "display": "block",
                        "marginBottom": "4px",
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
                            "border": "1px solid #d1d1d6",
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
            "borderBottom": "1px solid #e5e5e5",
        },
    )


def _build_status_section(status_messages: list) -> html.Div:
    return html.Div(
        [
            html.Div("Status", style={
                "fontSize": "13px",
                "fontWeight": "600",
                "color": "#1d1d1f",
                "marginBottom": "8px",
            }),
            html.Div(
                id="training-status",
                children=[
                    html.Div(msg, style={
                        "fontSize": "11px",
                        "color": "#86868b",
                        "fontFamily": "ui-monospace, monospace",
                        "lineHeight": "1.6",
                    })
                    for msg in status_messages
                ],
                style={
                    "padding": "8px",
                    "backgroundColor": "#f5f5f7",
                    "borderRadius": "6px",
                    "minHeight": "80px",
                    "maxHeight": "180px",
                    "overflowY": "auto",
                },
            ),
        ],
        style={"marginBottom": "24px"},
    )