from __future__ import annotations

"""Configuration page built from metadata to stay in sync with SSVAE options."""

from typing import Any, Dict, List, Optional, Sequence

import dash
from dash import Dash, Input, Output, State, dcc, html
import dash_bootstrap_components as dbc

from use_cases.dashboard.core import state as dashboard_state
from use_cases.dashboard.core.commands import UpdateModelConfigCommand
from use_cases.dashboard.core.config_metadata import (
    FieldSpec,
    SectionSpec,
    build_updates,
    default_config_dict,
    extract_initial_values,
    get_field_specs,
    get_modifiable_field_specs,
    get_section_specs,
    is_structural_parameter,
)
from use_cases.dashboard.core.state import _append_status_message

# Use only modifiable (non-structural) parameters for the configuration page
MODIFIABLE_FIELD_SPECS: Sequence[FieldSpec] = get_modifiable_field_specs()
SECTION_SPECS: Sequence[SectionSpec] = get_section_specs()
FIELD_OUTPUTS = [Output(spec.component_id, "value") for spec in MODIFIABLE_FIELD_SPECS]
FIELD_STATES = [State(spec.component_id, "value") for spec in MODIFIABLE_FIELD_SPECS]

_INPUT_STYLE = {
    "width": "100%",
    "padding": "10px 12px",
    "fontSize": "14px",
    "border": "1px solid #C6C6C6",
    "borderRadius": "6px",
    "fontFamily": "ui-monospace, monospace",
}
_DROPDOWN_STYLE = {
    "width": "100%",
    "fontSize": "14px",
    "fontFamily": "'Open Sans', Verdana, sans-serif",
}
_LABEL_STYLE = {
    "fontSize": "14px",
    "fontWeight": "600",
    "marginBottom": "6px",
    "color": "#000000",
    "fontFamily": "'Open Sans', Verdana, sans-serif",
}
_DESC_STYLE = {
    "fontSize": "13px",
    "color": "#6F6F6F",
    "marginBottom": "12px",
    "fontFamily": "'Open Sans', Verdana, sans-serif",
}


def _render_control(spec: FieldSpec):
    props = dict(spec.props)
    if spec.control == "number":
        return dcc.Input(
            id=spec.component_id,
            type="number",
            value=spec.default,
            debounce=True,
            style=_INPUT_STYLE,
            **props,
        )
    if spec.control == "text":
        return dcc.Input(
            id=spec.component_id,
            type="text",
            value=spec.default,
            debounce=True,
            style=_INPUT_STYLE,
            **props,
        )
    if spec.control == "dropdown":
        options = [opt.to_dash_option() for opt in spec.options]
        return dcc.Dropdown(
            id=spec.component_id,
            options=options,
            value=spec.default,
            clearable=False,
            style=_DROPDOWN_STYLE,
            **props,
        )
    if spec.control == "radio":
        options = [opt.to_dash_option() for opt in spec.options]
        return dbc.RadioItems(
            id=spec.component_id,
            options=options,
            value=spec.default,
            inline=False,
            style={"fontSize": "14px", "fontFamily": "'Open Sans', Verdana, sans-serif"},
        )
    if spec.control == "switch":
        return dbc.Switch(
            id=spec.component_id,
            value=bool(spec.default),
            style={"marginTop": "4px"},
            **props,
        )
    raise ValueError(f"Unsupported control type: {spec.control}")


def _render_field(spec: FieldSpec) -> html.Div:
    return html.Div(
        [
            html.Label(spec.label, style=_LABEL_STYLE, htmlFor=spec.component_id),
            html.Div(spec.description, style=_DESC_STYLE),
            _render_control(spec),
        ],
        style={"marginBottom": "12px"},
    )


def _build_architecture_summary(config: Any) -> html.Div:
    """Build read-only architecture summary showing locked structural parameters.

    Args:
        config: SSVAEConfig instance

    Returns:
        Div containing the architecture summary
    """
    # Prior description
    if config.prior_type == "standard":
        prior_desc = "Standard N(0,I)"
    elif config.prior_type in ["mixture", "vamp", "geometric_mog"]:
        prior_name = config.prior_type.replace("_", " ").title()
        prior_desc = f"{prior_name} ({config.num_components} components)"
    else:
        prior_desc = config.prior_type

    # Encoder description
    encoder_desc = "Convolutional" if config.encoder_type == "conv" else "Dense (MLP)"

    # Decoder flags
    component_aware = "Yes" if config.use_component_aware_decoder else "No"
    heteroscedastic = "Yes" if config.use_heteroscedastic_decoder else "No"

    # Reconstruction loss display
    recon_loss_desc = config.reconstruction_loss.upper()

    def _build_summary_row(label: str, value: str) -> html.Div:
        """Build a single summary row."""
        return html.Div([
            html.Span(label, style={
                "fontSize": "13px",
                "color": "#6F6F6F",
                "marginRight": "8px",
                "fontFamily": "'Open Sans', Verdana, sans-serif",
            }),
            html.Span(value, style={
                "fontSize": "13px",
                "color": "#000000",
                "fontWeight": "600",
                "fontFamily": "'Open Sans', Verdana, sans-serif",
            }),
        ], style={"marginBottom": "6px"})

    return html.Div([
        html.Div("Architecture 🔒 (fixed at creation)", style={
            "fontSize": "14px",
            "fontWeight": "700",
            "color": "#000000",
            "marginBottom": "12px",
            "fontFamily": "'Open Sans', Verdana, sans-serif",
        }),
        html.Div([
            _build_summary_row("Prior:", prior_desc),
            _build_summary_row("Encoder:", encoder_desc),
            _build_summary_row("Latent Dim:", str(config.latent_dim)),
            _build_summary_row("Recon Loss:", recon_loss_desc),
            html.Div(style={"height": "8px"}),  # Spacer
            _build_summary_row("Component-aware decoder:", component_aware),
            _build_summary_row("Heteroscedastic decoder:", heteroscedastic),
        ]),
    ], style={
        "padding": "16px",
        "backgroundColor": "#fafafa",
        "border": "1px solid #E6E6E6",
        "borderRadius": "6px",
        "marginBottom": "24px",
    })


def _build_section_tab(section: SectionSpec) -> dbc.Tab:
    section_fields = [spec for spec in MODIFIABLE_FIELD_SPECS if spec.section == section.id]
    rows: List[dbc.Row] = []
    current_cols: List[dbc.Col] = []
    width_acc = 0
    for spec in section_fields:
        col_width = min(max(spec.width, 1), 12)
        current_cols.append(dbc.Col(_render_field(spec), width=col_width))
        width_acc += col_width
        if width_acc >= 12:
            rows.append(dbc.Row(current_cols, class_name="gy-3"))
            current_cols = []
            width_acc = 0
    if current_cols:
        rows.append(dbc.Row(current_cols, class_name="gy-3"))

    content = html.Div(
        [
            html.P(section.description, style={"fontSize": "13px", "color": "#6F6F6F"}),
            *rows,
        ],
        className="p-4",
    )
    return dbc.Tab(content, tab_id=section.id, label=section.title)


def build_training_config_page(model_id: Optional[str] = None) -> html.Div:
    target_href = f"/model/{model_id}" if model_id else "/"
    tabs = [_build_section_tab(section) for section in SECTION_SPECS]

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H1(
                                "Model Configuration",
                                style={
                                    "fontSize": "24px",
                                    "fontWeight": "700",
                                    "color": "#000000",
                                    "margin": "0",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                            html.Div(
                                f"Active model: {model_id}" if model_id else "",
                                style={
                                    "fontSize": "13px",
                                    "color": "#6F6F6F",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                    "marginTop": "4px",
                                },
                            ),
                        ],
                    ),
                    html.Div(
                        [
                            dcc.Link(
                                dbc.Button(
                                    "Cancel",
                                    id="config-cancel-btn",
                                    color="light",
                                    style={
                                        "marginRight": "8px",
                                        "border": "1px solid #C6C6C6",
                                        "fontWeight": "600",
                                    },
                                ),
                                href=target_href,
                                style={"textDecoration": "none"},
                            ),
                            dbc.Button(
                                "Save Changes",
                                id="config-save-btn",
                                color="danger",
                                style={"fontWeight": "700"},
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center"},
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                    "padding": "24px 48px 16px 48px",
                    "backgroundColor": "#ffffff",
                },
            ),
            html.Div(
                [
                    # Architecture summary (populated via callback)
                    html.Div(
                        id="config-architecture-summary",
                        style={"padding": "24px 48px 0 48px"},
                    ),
                    dbc.Tabs(tabs, active_tab=SECTION_SPECS[0].id, class_name="px-3"),
                    html.Div(id="config-feedback", style={"padding": "16px 32px 32px 32px"}),
                ],
                style={"backgroundColor": "#fafafa"},
            ),
        ],
        style={"minHeight": "100vh", "backgroundColor": "#fafafa"},
    )


def register_config_page_callbacks(app: Dash) -> None:
    @app.callback(
        Output("config-architecture-summary", "children"),
        Input("training-config-store", "data"),
    )
    def populate_architecture_summary(config_dict: Optional[Dict[str, Any]]) -> html.Div:
        """Display architecture summary from config store."""
        if not config_dict:
            return html.Div()

        # Build a simple config object from the dict
        from types import SimpleNamespace
        config = SimpleNamespace(**config_dict)

        return _build_architecture_summary(config)

    @app.callback(
        FIELD_OUTPUTS,
        Input("training-config-store", "data"),
    )
    def populate_form_from_config(config_dict: Optional[Dict[str, Any]]) -> List[Any]:
        if not config_dict:
            return extract_initial_values(default_config_dict())
        return extract_initial_values(config_dict)

    @app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Output("config-feedback", "children"),
        Input("config-save-btn", "n_clicks"),
        *FIELD_STATES,
        State("training-config-store", "data"),
        prevent_initial_call=True,
    )
    def save_training_config(n_clicks: int, *args) -> tuple:
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        *field_values, config_store = args
        config_store = config_store or {}
        current_config = default_config_dict()
        current_config.update(config_store)

        try:
            updates = build_updates(field_values, current_config)
        except ValueError as exc:  # type: ignore[catching-class-any]
            error = html.Div(str(exc), style={"color": "#C10A27", "fontWeight": "600"})
            return dash.no_update, error

        if not updates:
            info = html.Div("No changes detected.", style={"color": "#6F6F6F"})
            return dash.no_update, info

        command = UpdateModelConfigCommand(updates=updates)
        success, message = dashboard_state.state_manager.dispatcher.execute(command)

        if not success:
            error_msg = html.Div(message, style={"color": "#C10A27", "fontWeight": "600"})
            return dash.no_update, error_msg

        _append_status_message(message)

        model_id = config_store.get("_model_id")
        if not model_id:
            with dashboard_state.state_manager.state_lock:
                if dashboard_state.state_manager.state.active_model:
                    model_id = dashboard_state.state_manager.state.active_model.model_id
        redirect_path = f"/model/{model_id}" if model_id else "/"
        success_msg = html.Div("Configuration saved.", style={"color": "#45717A"})
        return redirect_path, success_msg
