"""Home page - Model selection and creation."""

from __future__ import annotations

from dash import dcc, html
import dash_bootstrap_components as dbc

from use_cases.dashboard.core import state as dashboard_state
from use_cases.dashboard.core.model_runs import load_run_records


def build_home_layout() -> html.Div:
    """Build the home page with model cards."""
    dashboard_state.initialize_app_state()

    with dashboard_state.state_manager.state_lock:
        models = dashboard_state.state_manager.state.models if dashboard_state.state_manager.state else {}

    if not models:
        return _build_empty_state()

    model_cards = [
        _build_model_card(metadata)
        for metadata in sorted(models.values(), key=lambda m: m.last_modified, reverse=True)
    ]

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Link(
                                html.Img(
                                    src="/assets/infoteam_logo_basic.png",
                                    alt="infoteam software",
                                    style={"height": "50px", "width": "auto"},
                                ),
                                href="/",
                                style={"textDecoration": "none", "display": "block"},
                            ),
                        ],
                        style={"display": "inline-block", "marginRight": "32px"},
                    ),
                    html.Div(
                        [
                            html.H1(
                                "SSVAE Research Hub",
                                style={
                                    "fontSize": "28px",
                                    "fontWeight": "700",
                                    "margin": "0",
                                    "color": "#000000",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                            html.Div(
                                "Manage your semi-supervised learning experiments",
                                style={
                                    "fontSize": "15px",
                                    "color": "#6F6F6F",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                        ],
                        style={"display": "inline-block", "verticalAlign": "middle"},
                    ),
                ],
                style={"padding": "24px 48px", "backgroundColor": "#ffffff"},
            ),
            html.Div(style={"height": "4px", "backgroundColor": "#C10A27"}),
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
                    dcc.Link(
                        dbc.Button(
                            "View Experiment Runs",
                            color="light",
                            style={
                                "borderRadius": "8px",
                                "padding": "12px 24px",
                                "fontSize": "15px",
                                "fontWeight": "600",
                                "fontFamily": "'Open Sans', Verdana, sans-serif",
                            },
                        ),
                        href="/experiments",
                        style={"textDecoration": "none"},
                    ),
                ],
                style={
                    "padding": "24px 48px",
                    "backgroundColor": "#f5f5f5",
                    "display": "flex",
                    "gap": "12px",
                },
            ),
            html.Div(
                id="home-delete-feedback",
                style={"padding": "0 48px", "backgroundColor": "#fafafa"},
            ),
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
                            dcc.Link(
                                html.Img(
                                    src="/assets/infoteam_logo_basic.png",
                                    alt="infoteam software",
                                    style={"height": "50px", "width": "auto"},
                                ),
                                href="/",
                                style={"textDecoration": "none", "display": "block"},
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
                    html.Div(
                        dcc.Link(
                            "Browse existing experiment outputs",
                            href="/experiments",
                            style={
                                "display": "inline-block",
                                "marginTop": "16px",
                                "fontSize": "14px",
                                "fontWeight": "600",
                                "color": "#C10A27",
                                "textDecoration": "none",
                            },
                        ),
                    ),
                ],
                style={
                    "textAlign": "center",
                    "padding": "120px 48px",
                    "backgroundColor": "#fafafa",
                },
            ),
            
            # Delete feedback placeholder (keeps callbacks happy)
            html.Div(
                id="home-delete-feedback",
                style={
                    "padding": "0 48px",
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

    run_records = load_run_records(metadata.model_id)
    run_count = len(run_records)
    if run_records:
        try:
            last_run_dt = datetime.fromisoformat(run_records[0].timestamp)
            now = datetime.utcnow()
            delta = now - last_run_dt
            if delta.days > 0:
                last_run = f"{delta.days}d ago"
            elif delta.seconds > 3600:
                last_run = f"{delta.seconds // 3600}h ago"
            elif delta.seconds > 60:
                last_run = f"{delta.seconds // 60}m ago"
            else:
                last_run = "just now"
        except Exception:
            last_run = "recent"
    else:
        last_run = "none yet"
    
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
                        f"{metadata.dataset.upper()} • {metadata.labeled_count}/{metadata.dataset_total_samples} labeled • {metadata.total_epochs} epochs",
                        style={
                            "fontSize": "13px",
                            "color": "#6F6F6F",
                            "marginBottom": "12px",
                            "fontFamily": "'Open Sans', Verdana, sans-serif",
                        },
                    ),
                    html.Div(
                        f"Runs: {run_count} • last {last_run}",
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
                    dcc.Link(
                        dbc.Button(
                            "History",
                            color="light",
                            n_clicks=0,
                            style={
                                "border": "1px solid #C6C6C6",
                                "borderRadius": "6px",
                                "padding": "8px 16px",
                                "fontSize": "14px",
                                "fontWeight": "600",
                                "color": "#45717A",
                                "marginRight": "8px",
                                "fontFamily": "'Open Sans', Verdana, sans-serif",
                            },
                        ),
                        href=f"/experiments?model={metadata.model_id}",
                        style={"textDecoration": "none"},
                    ),
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
    """Modal for creating new model with full architecture configuration."""

    label_style = {
        "fontSize": "14px",
        "fontWeight": "600",
        "marginBottom": "6px",
        "display": "block",
        "fontFamily": "'Open Sans', Verdana, sans-serif",
    }

    input_style = {
        "width": "100%",
        "padding": "10px 12px",
        "fontSize": "14px",
        "border": "1px solid #C6C6C6",
        "borderRadius": "6px",
        "fontFamily": "ui-monospace, monospace",
    }

    text_input_style = input_style.copy()
    text_input_style["fontFamily"] = "'Open Sans', Verdana, sans-serif"

    help_style = {
        "fontSize": "12px",
        "color": "#6F6F6F",
        "marginTop": "6px",
        "fontFamily": "'Open Sans', Verdana, sans-serif",
    }

    section_title_style = {
        "fontSize": "16px",
        "fontWeight": "700",
        "color": "#000000",
        "marginTop": "24px",
        "marginBottom": "16px",
        "fontFamily": "'Open Sans', Verdana, sans-serif",
        "borderBottom": "2px solid #C10A27",
        "paddingBottom": "8px",
    }

    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Create New Model")),
            dbc.ModalBody(
                [
                    # Dataset Configuration Section
                    html.Div("Dataset Configuration", style=section_title_style),

                    html.Div(
                        [
                            html.Label("Model Name (optional)", style=label_style),
                            dcc.Input(
                                id="home-model-name-input",
                                type="text",
                                placeholder="e.g., Mixture VAE Baseline",
                                style=text_input_style,
                            ),
                        ],
                        style={"marginBottom": "16px"},
                    ),

                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Total Samples", style=label_style),
                                    dcc.Input(
                                        id="home-num-samples-input",
                                        type="number",
                                        min=32,
                                        max=70000,
                                        step=32,
                                        value=1024,
                                        placeholder="1024",
                                        style=input_style,
                                    ),
                                ],
                                style={"flex": "1", "marginRight": "12px"},
                            ),
                            html.Div(
                                [
                                    html.Label("Labeled Samples", style=label_style),
                                    dcc.Input(
                                        id="home-num-labeled-input",
                                        type="number",
                                        min=0,
                                        max=70000,
                                        step=1,
                                        value=128,
                                        placeholder="128",
                                        style=input_style,
                                    ),
                                ],
                                style={"flex": "1", "marginRight": "12px"},
                            ),
                            html.Div(
                                [
                                    html.Label("Sampling Seed", style=label_style),
                                    dcc.Input(
                                        id="home-seed-input",
                                        type="number",
                                        step=1,
                                        value=0,
                                        placeholder="0",
                                        style=input_style,
                                    ),
                                ],
                                style={"flex": "1"},
                            ),
                        ],
                        style={"display": "flex", "marginBottom": "8px"},
                    ),

                    html.Div(
                        id="home-unlabeled-preview",
                        style=help_style,
                    ),

                    # Model Architecture Section
                    html.Div("Model Architecture", style=section_title_style),

                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Encoder Type", style=label_style),
                                    dcc.Dropdown(
                                        id="home-encoder-type",
                                        options=[
                                            {"label": "Dense (MLP)", "value": "dense"},
                                            {"label": "Convolutional", "value": "conv"},
                                        ],
                                        value="conv",
                                        clearable=False,
                                        style={"fontSize": "14px"},
                                    ),
                                    html.Div("Conv recommended for MNIST", style=help_style),
                                ],
                                style={"flex": "1", "marginRight": "12px"},
                            ),
                            html.Div(
                                [
                                    html.Label("Latent Dimension", style=label_style),
                                    dcc.Input(
                                        id="home-latent-dim",
                                        type="number",
                                        min=2,
                                        max=256,
                                        value=2,
                                        placeholder="2",
                                        style=input_style,
                                    ),
                                    html.Div("Use 2 for visualization", style=help_style),
                                ],
                                style={"flex": "1"},
                            ),
                        ],
                        style={"display": "flex", "marginBottom": "16px"},
                    ),

                    # Hidden layers (conditional - only for Dense encoder)
                    html.Div(
                        [
                            html.Label("Hidden Layers (Dense encoder only)", style=label_style),
                            dcc.Input(
                                id="home-hidden-dims",
                                type="text",
                                value="256,128,64",
                                placeholder="256,128,64",
                                style=text_input_style,
                            ),
                            html.Div("Comma-separated layer sizes", style=help_style),
                        ],
                        id="home-hidden-dims-container",
                        style={"marginBottom": "16px", "display": "none"},  # Hidden by default
                    ),

                    # Prior Configuration Section
                    html.Div("Prior Configuration", style=section_title_style),

                    html.Div(
                        [
                            html.Label("Prior Type", style=label_style),
                            dcc.Dropdown(
                                id="home-prior-type",
                                options=[
                                    {"label": "Standard (Gaussian N(0,I))", "value": "standard"},
                                    {"label": "Mixture (Component-based)", "value": "mixture"},
                                    {"label": "VampPrior (Learned pseudo-inputs)", "value": "vamp"},
                                    {"label": "Geometric MoG (Fixed structure)", "value": "geometric_mog"},
                                ],
                                value="standard",
                                clearable=False,
                                style={"fontSize": "14px"},
                            ),
                            html.Div(
                                id="home-prior-help",
                                children="Simple Gaussian prior (no mixture structure)",
                                style=help_style,
                            ),
                        ],
                        style={"marginBottom": "16px"},
                    ),

                    # Mixture-specific options (conditional)
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Number of Components", style=label_style),
                                    dcc.Input(
                                        id="home-num-components",
                                        type="number",
                                        min=1,
                                        max=64,
                                        value=10,
                                        placeholder="10",
                                        style=input_style,
                                    ),
                                    html.Div("Should be ≥ num_classes (10)", style=help_style),
                                ],
                                style={"flex": "1", "marginRight": "12px"},
                            ),
                            html.Div(
                                [
                                    html.Label("Component Embedding Dim (optional)", style=label_style),
                                    dcc.Input(
                                        id="home-component-embedding-dim",
                                        type="number",
                                        min=1,
                                        max=128,
                                        value="",
                                        placeholder="(defaults to latent_dim)",
                                        style=input_style,
                                    ),
                                    html.Div("Leave blank to use latent_dim", style=help_style),
                                ],
                                style={"flex": "1"},
                            ),
                        ],
                        id="home-mixture-options",
                        style={"display": "none", "marginBottom": "16px"},  # Hidden by default
                    ),

                    html.Div(
                        [
                            dbc.Checklist(
                                id="home-use-component-aware-decoder",
                                options=[
                                    {"label": " Use Component-Aware Decoder (separate pathways per component)", "value": True}
                                ],
                                value=[True],
                                style={"fontSize": "14px", "fontFamily": "'Open Sans', Verdana, sans-serif"},
                            ),
                            html.Div("Recommended for mixture priors", style=help_style),
                        ],
                        id="home-component-aware-container",
                        style={"display": "none", "marginBottom": "16px"},  # Hidden by default
                    ),

                    html.Div(
                        id="home-create-feedback",
                        style={"fontSize": "14px", "marginTop": "16px", "color": "#C10A27"},
                    ),
                ],
                style={"maxHeight": "70vh", "overflowY": "auto"},
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
        size="lg",  # Larger modal for more content
    )
