from __future__ import annotations

"""Helpers to provide Dash with a validation layout covering all callback IDs."""

from dash import dcc, html
import dash_bootstrap_components as dbc

from use_cases.dashboard.core.config_metadata import get_field_specs


def _field_stub_components():
    components = []
    for spec in get_field_specs():
        if spec.control == "number":
            components.append(dcc.Input(id=spec.component_id, type="number"))
        elif spec.control == "text":
            components.append(dcc.Input(id=spec.component_id, type="text"))
        elif spec.control == "dropdown":
            options = [opt.to_dash_option() for opt in spec.options] or [
                {"label": "option", "value": spec.default or "option"}
            ]
            components.append(
                dcc.Dropdown(
                    id=spec.component_id,
                    options=options,
                    value=options[0]["value"],
                )
            )
        elif spec.control == "radio":
            options = [opt.to_dash_option() for opt in spec.options] or [
                {"label": "option", "value": spec.default or "option"}
            ]
            components.append(
                dbc.RadioItems(
                    id=spec.component_id,
                    options=options,
                    value=options[0]["value"],
                )
            )
        elif spec.control == "switch":
            components.append(dbc.Switch(id=spec.component_id, value=bool(spec.default)))
    return components


def build_validation_layout() -> html.Div:
    """Return invisible components so Dash validates callbacks across pages."""
    field_components = _field_stub_components()

    shared_components = [
        dcc.Store(id="selected-sample-store"),
        dcc.Store(id="labels-store"),
        dcc.Store(id="training-control-store"),
        dcc.Store(id="latent-store"),
        dcc.Interval(id="training-poll"),
        dcc.Store(id="keyboard-label-store"),
        dcc.Interval(id="keyboard-poll"),
        dcc.Interval(id="resize-setup-trigger"),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("stub")),
                dbc.ModalBody(html.Div(id="modal-training-info")),
                dbc.ModalFooter(
                    [
                        dbc.Button("confirm", id="modal-confirm-button"),
                        dbc.Button("cancel", id="modal-cancel-button"),
                    ]
                ),
            ],
            id="training-confirm-modal",
        ),
        dbc.Button("start", id="start-training-button"),
        dcc.Input(id="num-epochs-input", type="number"),
        html.Div(id="training-status"),
        dcc.Graph(id="latent-scatter"),
        html.Div(id="scatter-legend"),
        dbc.RadioItems(id="color-mode-radio", options=[{"label": "stub", "value": "user_labels"}]),
        dcc.Graph(id="loss-curves"),
        html.Div(id="selected-sample-header"),
        html.Img(id="original-image"),
        html.Img(id="reconstructed-image"),
        html.Div(id="prediction-info"),
        html.Div(id="label-feedback"),
        html.Div(id="dataset-stats"),
        dbc.Button("train hub start", id="training-hub-start-button"),
        dbc.Button("train hub stop", id="training-hub-stop-button"),
        dcc.Input(id="training-hub-epochs", type="number"),
        dcc.Input(id="training-hub-lr", type="number"),
        dcc.Input(id="training-hub-recon", type="number"),
        dcc.Input(id="training-hub-kl", type="number"),
        dcc.Interval(id="training-hub-poll"),
        dcc.Store(id="training-hub-control-store"),
        dcc.Store(id="training-hub-latent-store"),
        dcc.Graph(id="training-hub-loss-curves"),
        html.Pre(id="training-hub-terminal"),
        html.Div(id="training-hub-status-metrics"),
        html.Div(id="training-hub-status-text"),
        html.Div(id="training-hub-params-content"),
        dbc.Button("toggle", id="training-hub-params-toggle"),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("stub")),
                dbc.ModalBody(html.Div(id="training-hub-modal-info")),
            ],
            id="training-hub-modal",
        ),
        dcc.Download(id="training-hub-terminal-download"),
        html.Div(id="home-create-feedback"),
        html.Div(id="home-delete-feedback"),
        html.Div(id="home-unlabeled-preview"),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("create")),
                dbc.ModalBody(
                    [
                        dcc.Input(id="home-model-name-input", type="text"),
                        dcc.Input(id="home-num-samples-input", type="number"),
                        dcc.Input(id="home-num-labeled-input", type="number"),
                        dcc.Input(id="home-seed-input", type="number"),
                    ]
                ),
            ],
            id="home-create-modal",
        ),
        html.Div(id="run-history-list"),
        dcc.Location(id="experiments-url"),
        dcc.Store(id="experiments-run-data"),
        dcc.Store(id="experiments-model-list"),
        dcc.Dropdown(id="experiments-tag-filter", options=[], value=None),
        dbc.Button("Refresh", id="experiments-refresh-btn"),
        html.Div(id="experiments-model-list-ui"),
        html.Div(id="experiments-run-list"),
        html.Div(id="experiments-run-detail"),
        html.Div(id="experiments-filter-indicator"),
        html.Div(id="config-feedback"),
    ]

    return html.Div(field_components + shared_components, style={"display": "none"})
