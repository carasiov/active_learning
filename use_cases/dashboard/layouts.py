"""Layout construction for the SSVAE dashboard - IMPROVED VERSION."""

from __future__ import annotations

from dash import dcc, html
import dash_bootstrap_components as dbc
import numpy as np

from use_cases.dashboard.state import app_state, state_lock, initialize_model_and_data


def build_dashboard_layout() -> html.Div:
    """Build and return the Dash layout for the dashboard UI."""
    initialize_model_and_data()
    with state_lock:
        config = app_state["config"]
        recon_weight_value = float(np.clip(config.recon_weight, 0.0, 5000.0))
        kl_weight_value = float(np.clip(config.kl_weight, 0.0, 1.0))
        learning_rate_value = float(np.clip(config.learning_rate, 0.0001, 0.01))
        default_epochs = max(1, int(app_state["training"]["target_epochs"] or 5))
        latent_version = int(app_state["ui"]["latent_version"])
        existing_status = list(app_state["training"]["status_messages"])

    status_initial_children = (
        html.Ul([html.Li(msg) for msg in existing_status], className="mb-0 small")
        if existing_status
        else html.Span("Idle.", className="text-muted")
    )

    return dbc.Container(
        [
            # Hidden stores and intervals
            dcc.Store(id="selected-sample-store", data=int(app_state["ui"]["selected_sample"])),
            dcc.Store(id="labels-store", data={"version": int(app_state["ui"]["labels_version"])}),
            dcc.Store(id="training-control-store", data={"token": 0}),
            dcc.Store(id="latent-store", data={"version": latent_version}),
            dcc.Interval(id="training-poll", interval=2000, n_intervals=0, disabled=True),
            dcc.Store(id="keyboard-label-store"),
            dcc.Interval(id="keyboard-poll", interval=300, n_intervals=0, disabled=False),
            
            # Header
            dbc.Row(
                dbc.Col(
                    html.H1("SSVAE Active Learning Dashboard", className="mb-1"),
                    width=12,
                ),
                className="mb-3 mt-2",
            ),
            
            # Top Section: Training Progress + Dataset Stats side by side
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(html.Strong("Training Progress")),
                                dbc.CardBody(
                                    dcc.Graph(id="loss-curves", style={"height": "320px"}),
                                    className="p-2",
                                ),
                            ],
                            className="shadow-sm",
                        ),
                        width=8,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(html.Strong("Dataset Statistics")),
                                dbc.CardBody(
                                    html.Div(id="dataset-stats", className="p-2"),
                                    style={"minHeight": "320px"},
                                ),
                            ],
                            className="shadow-sm",
                        ),
                        width=4,
                    ),
                ],
                className="mb-3",
            ),
            
            # Main Content: Latent Space + Sample Inspector
            dbc.Row(
                [
                    # Left: Latent Space Visualization
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    html.Strong("Latent Space Visualization"),
                                                    width="auto",
                                                ),
                                                dbc.Col(
                                                    dbc.RadioItems(
                                                        id="color-mode-radio",
                                                        options=[
                                                            {"label": "User Labels", "value": "user_labels"},
                                                            {"label": "Predicted", "value": "pred_class"},
                                                            {"label": "True Label", "value": "true_class"},
                                                            {"label": "Certainty", "value": "certainty"},
                                                        ],
                                                        value="user_labels",
                                                        inline=True,
                                                        className="ms-3",
                                                    ),
                                                    width=True,
                                                ),
                                            ],
                                            align="center",
                                        ),
                                    ),
                                    dbc.CardBody(
                                        dcc.Graph(
                                            id="latent-scatter",
                                            style={"height": "600px"},
                                            config={"displayModeBar": True, "scrollZoom": True, "responsive": True},
                                        ),
                                        className="p-2",
                                    ),
                                ],
                                className="shadow-sm",
                            ),
                        ],
                        width=8,
                    ),
                    
                    # Right: Sample Inspector + Training Controls
                    dbc.Col(
                        [
                            # Sample Inspector Card
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5(id="selected-sample-header", className="mb-0"),
                                    ),
                                    dbc.CardBody(
                                        [
                                            # Images
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Div("Original", className="text-center fw-bold mb-2"),
                                                            html.Div(
                                                                html.Img(
                                                                    id="original-image",
                                                                    style={"width": "100%", "maxWidth": "200px"},
                                                                ),
                                                                className="text-center",
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Div("Reconstruction", className="text-center fw-bold mb-2"),
                                                            html.Div(
                                                                html.Img(
                                                                    id="reconstructed-image",
                                                                    style={"width": "100%", "maxWidth": "200px"},
                                                                ),
                                                                className="text-center",
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            
                                            # Prediction Info
                                            html.Div(
                                                id="prediction-info",
                                                className="p-2 mb-3 bg-light rounded text-center",
                                                style={"fontSize": "0.9rem"},
                                            ),
                                            
                                            # Labeling Section
                                            html.Div(
                                                [
                                                    html.Div("Assign Label", className="fw-bold mb-2"),
                                                    html.Div(
                                                        [
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        dbc.Button(
                                                                            str(digit),
                                                                            id={"type": "label-button", "label": digit},
                                                                            color="primary",
                                                                            outline=True,
                                                                            n_clicks=0,
                                                                            style={"width": "100%"},
                                                                        ),
                                                                        width=2,
                                                                        className="mb-2 px-1",
                                                                    )
                                                                    for digit in range(5)
                                                                ],
                                                                className="g-1",
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        dbc.Button(
                                                                            str(digit),
                                                                            id={"type": "label-button", "label": digit},
                                                                            color="primary",
                                                                            outline=True,
                                                                            n_clicks=0,
                                                                            style={"width": "100%"},
                                                                        ),
                                                                        width=2,
                                                                        className="mb-2 px-1",
                                                                    )
                                                                    for digit in range(5, 10)
                                                                ],
                                                                className="g-1",
                                                            ),
                                                            dbc.Button(
                                                                "Delete Label",
                                                                id="delete-label-button",
                                                                color="danger",
                                                                outline=True,
                                                                n_clicks=0,
                                                                className="w-100 mt-2",
                                                            ),
                                                        ],
                                                    ),
                                                    html.Div(
                                                        id="label-feedback",
                                                        className="text-muted small mt-2 text-center",
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                                className="shadow-sm mb-3",
                            ),
                            
                            # Training Controls Card (Compact)
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.Strong("Training Controls"),
                                    ),
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Epochs", className="small mb-1"),
                                                            dcc.Input(
                                                                id="num-epochs-input",
                                                                type="number",
                                                                min=1,
                                                                max=200,
                                                                step=1,
                                                                value=default_epochs,
                                                                debounce=True,
                                                                style={"width": "100%"},
                                                            ),
                                                        ],
                                                        width=4,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("LR", className="small mb-1"),
                                                            dcc.Input(
                                                                id="learning-rate-slider",
                                                                type="number",
                                                                min=0.0001,
                                                                max=0.01,
                                                                step=0.0001,
                                                                value=learning_rate_value,
                                                                debounce=True,
                                                                style={"width": "100%"},
                                                            ),
                                                        ],
                                                        width=4,
                                                    ),
                                                    dbc.Col(
                                                        dbc.Button(
                                                            "Train",
                                                            id="start-training-button",
                                                            color="success",
                                                            className="w-100",
                                                            style={"marginTop": "23px"},
                                                            n_clicks=0,
                                                        ),
                                                        width=4,
                                                    ),
                                                ],
                                                className="mb-2",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Recon", className="small mb-1"),
                                                            dcc.Input(
                                                                id="recon-weight-slider",
                                                                type="number",
                                                                min=0,
                                                                max=5000,
                                                                step=50,
                                                                value=recon_weight_value,
                                                                debounce=True,
                                                                style={"width": "100%"},
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("KL", className="small mb-1"),
                                                            dcc.Input(
                                                                id="kl-weight-slider",
                                                                type="number",
                                                                min=0.0,
                                                                max=1.0,
                                                                step=0.01,
                                                                value=kl_weight_value,
                                                                debounce=True,
                                                                style={"width": "100%"},
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                ],
                                            ),
                                            html.Hr(className="my-2"),
                                            html.Div(
                                                html.Div(status_initial_children, id="training-status"),
                                                style={"maxHeight": "120px", "overflowY": "auto"},
                                            ),
                                        ],
                                        className="p-2",
                                    ),
                                ],
                                className="shadow-sm",
                            ),
                        ],
                        width=4,
                    ),
                ],
                className="mb-4",
            ),
        ],
        fluid=True,
        className="px-4 py-3",
        style={"backgroundColor": "#f8f9fa"},
    )