"""Home page - Model selection and creation."""

from __future__ import annotations

from dash import dcc, html
import dash_bootstrap_components as dbc

from use_cases.dashboard.core import state as dashboard_state


def build_home_layout() -> html.Div:
    """Build the home page with model cards."""
    dashboard_state.initialize_app_state()
    
    with dashboard_state.state_lock:
        models = dashboard_state.app_state.models if dashboard_state.app_state else {}
    
    # Empty state
    if not models:
        return _build_empty_state()
    
    # Model cards
    model_cards = []
    for metadata in sorted(models.values(), key=lambda m: m.last_modified, reverse=True):
        model_cards.append(_build_model_card(metadata))
    
    return html.Div(
        [
            # Header
            html.Div(
                [
                    html.Div(
                        [
                            html.Img(
                                src="/assets/infoteam_logo_basic.png",
                                alt="infoteam software",
                                style={"height": "50px", "width": "auto"},
                            ),
                        ],
                        style={"display": "inline-block", "marginRight": "32px"},
                    ),
                    html.Div(
                        [
                            html.H1("SSVAE Research Hub", style={
                                "fontSize": "28px",
                                "fontWeight": "700",
                                "margin": "0",
                                "color": "#000000",
                                "fontFamily": "'Open Sans', Verdana, sans-serif",
                            }),
                            html.Div("Manage your semi-supervised learning experiments", style={
                                "fontSize": "15px",
                                "color": "#6F6F6F",
                                "fontFamily": "'Open Sans', Verdana, sans-serif",
                            }),
                        ],
                        style={"display": "inline-block", "verticalAlign": "middle"},
                    ),
                ],
                style={
                    "padding": "24px 48px",
                    "backgroundColor": "#ffffff",
                },
            ),
            html.Div(style={"height": "4px", "backgroundColor": "#C10A27"}),
            
            # Action bar
            html.Div(
                [
                    dbc.Button(
                        "+ New Model",
                        id="home-new-model-btn",
                        n_clicks=0,
                        style={
                            "backgroundColor": "#C10A27",
                            "border": "none",
                            "borderRadius": "8px",
                            "padding": "12px 24px",
                            "fontSize": "15px",
                            "fontWeight": "700",
                            "color": "#ffffff",
                            "fontFamily": "'Open Sans', Verdana, sans-serif",
                        },
                    ),
                ],
                style={"padding": "24px 48px", "backgroundColor": "#f5f5f5"},
            ),
            
            # Model grid
            html.Div(
                model_cards,
                style={
                    "padding": "32px 48px",
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fill, minmax(400px, 1fr))",
                    "gap": "24px",
                    "backgroundColor": "#fafafa",
                },
            ),
            
            # Create model modal
            _build_create_modal(),
        ],
        style={
            "fontFamily": "'Open Sans', Verdana, sans-serif",
            "backgroundColor": "#fafafa",
            "minHeight": "100vh",
        },
    )


def _build_empty_state() -> html.Div:
    """Empty state when no models exist."""
    return html.Div(
        [
            # Header
            html.Div(
                [
                    html.Div(
                        [
                            html.Img(
                                src="/assets/infoteam_logo_basic.png",
                                alt="infoteam software",
                                style={"height": "50px", "width": "auto"},
                            ),
                        ],
                        style={"display": "inline-block", "marginRight": "32px"},
                    ),
                    html.Div(
                        [
                            html.H1("SSVAE Research Hub", style={
                                "fontSize": "28px",
                                "fontWeight": "700",
                                "margin": "0",
                                "color": "#000000",
                                "fontFamily": "'Open Sans', Verdana, sans-serif",
                            }),
                            html.Div("Manage your semi-supervised learning experiments", style={
                                "fontSize": "15px",
                                "color": "#6F6F6F",
                                "fontFamily": "'Open Sans', Verdana, sans-serif",
                            }),
                        ],
                        style={"display": "inline-block", "verticalAlign": "middle"},
                    ),
                ],
                style={
                    "padding": "24px 48px",
                    "backgroundColor": "#ffffff",
                },
            ),
            html.Div(style={"height": "4px", "backgroundColor": "#C10A27"}),
            
            # Empty state message
            html.Div(
                [
                    html.Div("🎯", style={"fontSize": "64px", "marginBottom": "24px"}),
                    html.H2("No Models Yet", style={
                        "fontSize": "24px",
                        "fontWeight": "700",
                        "color": "#000000",
                        "marginBottom": "12px",
                        "fontFamily": "'Open Sans', Verdana, sans-serif",
                    }),
                    html.P(
                        "Create your first model to start experimenting with semi-supervised learning.",
                        style={
                            "fontSize": "16px",
                            "color": "#6F6F6F",
                            "marginBottom": "32px",
                            "fontFamily": "'Open Sans', Verdana, sans-serif",
                        },
                    ),
                    dbc.Button(
                        "+ Create Your First Model",
                        id="home-new-model-btn",
                        n_clicks=0,
                        style={
                            "backgroundColor": "#C10A27",
                            "border": "none",
                            "borderRadius": "8px",
                            "padding": "16px 32px",
                            "fontSize": "16px",
                            "fontWeight": "700",
                            "color": "#ffffff",
                            "fontFamily": "'Open Sans', Verdana, sans-serif",
                        },
                    ),
                ],
                style={
                    "textAlign": "center",
                    "padding": "120px 48px",
                    "backgroundColor": "#fafafa",
                },
            ),
            
            # Create model modal
            _build_create_modal(),
        ],
        style={
            "fontFamily": "'Open Sans', Verdana, sans-serif",
            "backgroundColor": "#fafafa",
            "minHeight": "100vh",
        },
    )


def _build_model_card(metadata) -> html.Div:
    """Build a single model card."""
    from datetime import datetime
    
    # Format last modified
    try:
        last_mod = datetime.fromisoformat(metadata.last_modified)
        now = datetime.utcnow()
        delta = now - last_mod
        
        if delta.days > 0:
            time_ago = f"{delta.days}d ago"
        elif delta.seconds > 3600:
            time_ago = f"{delta.seconds // 3600}h ago"
        elif delta.seconds > 60:
            time_ago = f"{delta.seconds // 60}m ago"
        else:
            time_ago = "just now"
    except Exception:
        time_ago = "unknown"
    
    # Loss display
    loss_display = f"{metadata.latest_loss:.4f}" if metadata.latest_loss else "—"
    
    return html.Div(
        [
            html.Div(
                [
                    html.H3(metadata.name, style={
                        "fontSize": "18px",
                        "fontWeight": "700",
                        "color": "#000000",
                        "marginBottom": "4px",
                        "fontFamily": "'Open Sans', Verdana, sans-serif",
                    }),
                    html.Div(
                        f"📁 {metadata.model_id}",
                        style={
                            "fontSize": "12px",
                            "color": "#999999",
                            "marginBottom": "8px",
                            "fontFamily": "'Courier New', monospace",
                        },
                    ),
                    html.Div(
                        f"{metadata.dataset.upper()} • {metadata.labeled_count} labels • {metadata.total_epochs} epochs",
                        style={
                            "fontSize": "13px",
                            "color": "#6F6F6F",
                            "marginBottom": "12px",
                            "fontFamily": "'Open Sans', Verdana, sans-serif",
                        },
                    ),
                    html.Div(
                        [
                            html.Span(f"Last: {time_ago}", style={
                                "marginRight": "16px",
                                "fontSize": "13px",
                                "color": "#6F6F6F",
                                "fontFamily": "'Open Sans', Verdana, sans-serif",
                            }),
                            html.Span(f"Loss: {loss_display}", style={
                                "fontSize": "13px",
                                "color": "#6F6F6F",
                                "fontFamily": "'Open Sans', Verdana, sans-serif",
                            }),
                        ],
                    ),
                ],
                style={"flex": "1"},
            ),
            html.Div(
                [
                    dbc.Button(
                        "Open",
                        id={"type": "home-open-model", "model_id": metadata.model_id},
                        n_clicks=0,
                        style={
                            "backgroundColor": "#45717A",
                            "border": "none",
                            "borderRadius": "6px",
                            "padding": "8px 20px",
                            "fontSize": "14px",
                            "fontWeight": "600",
                            "color": "#ffffff",
                            "marginRight": "8px",
                            "fontFamily": "'Open Sans', Verdana, sans-serif",
                        },
                    ),
                    dbc.Button(
                        "Delete",
                        id={"type": "home-delete-model", "model_id": metadata.model_id},
                        n_clicks=0,
                        className="delete-button-stop-propagation",
                        style={
                            "backgroundColor": "#ffffff",
                            "border": "1px solid #C6C6C6",
                            "borderRadius": "6px",
                            "padding": "8px 16px",
                            "fontSize": "14px",
                            "fontWeight": "600",
                            "color": "#6F6F6F",
                            "fontFamily": "'Open Sans', Verdana, sans-serif",
                        },
                    ),
                ],
            ),
        ],
        style={
            "backgroundColor": "#ffffff",
            "border": "1px solid #C6C6C6",
            "borderRadius": "8px",
            "padding": "24px",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "space-between",
        },
    )


def _build_create_modal() -> dbc.Modal:
    """Modal for creating new model."""
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Create New Model")),
            dbc.ModalBody(
                [
                    html.Div(
                        [
                            html.Label("Model Name (optional)", style={
                                "fontSize": "14px",
                                "fontWeight": "600",
                                "marginBottom": "6px",
                                "display": "block",
                                "fontFamily": "'Open Sans', Verdana, sans-serif",
                            }),
                            dcc.Input(
                                id="home-model-name-input",
                                type="text",
                                placeholder="e.g., Baseline Experiment",
                                style={
                                    "width": "100%",
                                    "padding": "10px 12px",
                                    "fontSize": "14px",
                                    "border": "1px solid #C6C6C6",
                                    "borderRadius": "6px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                        ],
                        style={"marginBottom": "20px"},
                    ),
                    html.Div(
                        [
                            html.Label("Configuration Preset", style={
                                "fontSize": "14px",
                                "fontWeight": "600",
                                "marginBottom": "12px",
                                "display": "block",
                                "fontFamily": "'Open Sans', Verdana, sans-serif",
                            }),
                            dbc.RadioItems(
                                id="home-config-preset",
                                options=[
                                    {"label": "Default MNIST (balanced)", "value": "default"},
                                    {"label": "High Reconstruction (recon=5000)", "value": "high_recon"},
                                    {"label": "Classification Focus (label_weight=10)", "value": "classification"},
                                ],
                                value="default",
                                style={"fontSize": "14px", "fontFamily": "'Open Sans', Verdana, sans-serif"},
                            ),
                        ],
                        style={"marginBottom": "20px"},
                    ),
                    html.Div(
                        id="home-create-feedback",
                        style={"fontSize": "14px", "marginTop": "12px"},
                    ),
                ]
            ),
            dbc.ModalFooter(
                [
                    dbc.Button(
                        "Cancel",
                        id="home-cancel-create",
                        n_clicks=0,
                        style={
                            "backgroundColor": "#ffffff",
                            "border": "1px solid #C6C6C6",
                            "color": "#6F6F6F",
                            "borderRadius": "6px",
                            "padding": "8px 16px",
                            "fontSize": "14px",
                            "fontWeight": "600",
                            "fontFamily": "'Open Sans', Verdana, sans-serif",
                        },
                    ),
                    dbc.Button(
                        "Create Model",
                        id="home-confirm-create",
                        n_clicks=0,
                        style={
                            "backgroundColor": "#C10A27",
                            "border": "none",
                            "color": "#ffffff",
                            "marginLeft": "8px",
                            "borderRadius": "6px",
                            "padding": "8px 16px",
                            "fontSize": "14px",
                            "fontWeight": "700",
                            "fontFamily": "'Open Sans', Verdana, sans-serif",
                        },
                    ),
                ]
            ),
        ],
        id="home-create-modal",
        is_open=False,
        centered=True,
    )
