"""Layout construction for the SSVAE dashboard."""

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
            dcc.Store(id="selected-sample-store", data=int(app_state["ui"]["selected_sample"])),
            dcc.Store(id="labels-store", data={"version": int(app_state["ui"]["labels_version"])}),
            dcc.Store(id="training-control-store", data={"token": 0}),
            dcc.Store(id="latent-store", data={"version": latent_version}),
            dcc.Interval(id="training-poll", interval=2000, n_intervals=0, disabled=True),
            html.H1("SSVAE Active Learning Dashboard", className="mb-4"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Training Controls"),
                                dbc.CardBody(
                                    [
                                        dbc.Label("Reconstruction Weight"),
                                        dcc.Slider(
                                            id="recon-weight-slider",
                                            min=0,
                                            max=5000,
                                            step=50,
                                            value=recon_weight_value,
                                            marks={0: "0", 2500: "2500", 5000: "5000"},
                                            tooltip={"placement": "bottom", "always_visible": False},
                                        ),
                                        dbc.Label("KL Weight", className="mt-3"),
                                        dcc.Slider(
                                            id="kl-weight-slider",
                                            min=0.0,
                                            max=1.0,
                                            step=0.01,
                                            value=kl_weight_value,
                                            marks={0.0: "0", 0.5: "0.5", 1.0: "1"},
                                            tooltip={"placement": "bottom", "always_visible": False},
                                        ),
                                        dbc.Label("Learning Rate", className="mt-3"),
                                        dcc.Slider(
                                            id="learning-rate-slider",
                                            min=0.0001,
                                            max=0.01,
                                            step=0.0001,
                                            value=learning_rate_value,
                                            marks={0.0001: "1e-4", 0.005: "5e-3", 0.01: "1e-2"},
                                            tooltip={"placement": "bottom", "always_visible": False},
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Epochs", className="mt-3"),
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
                                                    md=4,
                                                ),
                                                dbc.Col(
                                                    dbc.Button(
                                                        "Start Training",
                                                        id="start-training-button",
                                                        color="success",
                                                        className="mt-4",
                                                        n_clicks=0,
                                                    ),
                                                    md="auto",
                                                ),
                                            ],
                                            className="g-3 align-items-end",
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        width=8,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Training Status"),
                                dbc.CardBody(
                                    html.Div(status_initial_children, id="training-status", className="m-0"),
                                ),
                            ],
                            className="mb-3",
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
                            dbc.Card(
                                [
                                    dbc.CardHeader("Color By"),
                                    dbc.CardBody(
                                        dbc.RadioItems(
                                            id="color-mode-radio",
                                            options=[
                                                {"label": "User Labels", "value": "user_labels"},
                                                {"label": "Predicted Class", "value": "pred_class"},
                                                {"label": "True Label", "value": "true_class"},
                                                {"label": "Certainty", "value": "certainty"},
                                            ],
                                            value="user_labels",
                                            inline=True,
                                        )
                                    ),
                                ],
                                className="mb-3",
                            ),
                            dcc.Graph(
                                id="latent-scatter",
                                style={"height": "650px"},
                                config={"displayModeBar": True, "scrollZoom": True, "responsive": True},
                            ),
                        ],
                        width=8,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H4(id="selected-sample-header", className="mb-3"),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.H6("Original"),
                                                    html.Img(
                                                        id="original-image",
                                                        style={"width": "100%", "maxWidth": "250px"},
                                                    ),
                                                ],
                                                width=6,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.H6("Reconstruction"),
                                                    html.Img(
                                                        id="reconstructed-image",
                                                        style={"width": "100%", "maxWidth": "250px"},
                                                    ),
                                                ],
                                                width=6,
                                            ),
                                        ],
                                        className="mb-3",
                                    ),
                                    html.Div(id="prediction-info", className="mb-3 fw-semibold"),
                                    html.Div(
                                        [
                                            dbc.Label("Assign Label"),
                                            dbc.ButtonGroup(
                                                [
                                                    dbc.Button(
                                                        str(digit),
                                                        id={"type": "label-button", "label": digit},
                                                        color="primary",
                                                        outline=True,
                                                        n_clicks=0,
                                                        size="sm",
                                                    )
                                                    for digit in range(10)
                                                ],
                                                className="flex-wrap gap-1",
                                            ),
                                            dbc.Button(
                                                "Delete Label",
                                                id="delete-label-button",
                                                color="danger",
                                                outline=True,
                                                n_clicks=0,
                                                size="sm",
                                                className="ms-2",
                                            ),
                                        ],
                                        className="mb-2",
                                    ),
                                    html.Div(id="label-feedback", className="text-muted"),
                                ]
                            ),
                        ],
                        width=4,
                    ),
                ]
            ),
        ],
        fluid=True,
        className="pt-3 pb-5",
    )
