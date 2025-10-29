"""Home page callbacks - model management."""

from __future__ import annotations

from dash import Dash, Input, Output, State, ALL, html, no_update
from dash.exceptions import PreventUpdate
import dash
import dash_bootstrap_components as dbc

from use_cases.dashboard.core import state as dashboard_state
from use_cases.dashboard.core.commands import CreateModelCommand, LoadModelCommand, DeleteModelCommand
from use_cases.dashboard.utils.callback_utils import logged_callback


def register_home_callbacks(app: Dash) -> None:
    """Register home page callbacks."""
    
    @app.callback(
        Output("home-create-modal", "is_open"),
        Output("home-model-name-input", "value"),
        Output("home-config-preset", "value"),
        Input("home-new-model-btn", "n_clicks"),
        Input("home-confirm-create", "n_clicks"),
        Input("home-cancel-create", "n_clicks"),
        State("home-create-modal", "is_open"),
        prevent_initial_call=True,
    )
    @logged_callback("toggle_create_modal")
    def toggle_create_modal(new_clicks, confirm_clicks, cancel_clicks, is_open):
        """Open/close create model modal."""
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if triggered_id == "home-new-model-btn":
            # Open modal and reset inputs
            return True, "", "default"
        elif triggered_id == "home-cancel-create":
            # Explicit cancel closes modal
            return False, "", "default"
        elif triggered_id == "home-confirm-create":
            # Keep modal open; create callback will navigate on success
            # or show feedback on failure
            return is_open, "", "default"
        
        raise PreventUpdate
    
    @app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Output("home-create-feedback", "children"),
        Input("home-confirm-create", "n_clicks"),
        State("home-model-name-input", "value"),
        State("home-config-preset", "value"),
        prevent_initial_call=True,
    )
    def create_model(n_clicks, model_name, preset):
        """Create new model and navigate to it."""
        if not n_clicks:
            raise PreventUpdate
        
        # Create model
        command = CreateModelCommand(
            name=model_name if model_name else None,
            config_preset=preset
        )
        success, model_id_or_error = dashboard_state.dispatcher.execute(command)
        
        if not success:
            return no_update, html.Div(model_id_or_error, style={"color": "#C10A27"})
        
        # model_id_or_error is actually the model_id on success
        model_id = model_id_or_error
        
        # Now load the model
        load_command = LoadModelCommand(model_id=model_id)
        load_success, load_message = dashboard_state.dispatcher.execute(load_command)
        
        if not load_success:
            return no_update, html.Div(f"Created but failed to load: {load_message}", style={"color": "#C10A27"})
        
        # Navigate to model
        return f"/model/{model_id}", ""
    
    @app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Input({"type": "home-open-model", "model_id": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    @logged_callback("open_model")
    def open_model(n_clicks_list):
        """Navigate to model dashboard."""
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        # Find which button was clicked
        triggered_id = ctx.triggered[0]["prop_id"]
        if ".n_clicks" not in triggered_id:
            raise PreventUpdate
        
        # Extract model_id from triggered_id JSON
        import json
        try:
            id_dict = json.loads(triggered_id.split(".")[0])
            model_id = id_dict["model_id"]
        except (json.JSONDecodeError, KeyError) as e:
            raise PreventUpdate
        
        # Load model
        command = LoadModelCommand(model_id=model_id)
        success, message = dashboard_state.dispatcher.execute(command)
        
        if not success:
            raise PreventUpdate
        
        # Navigate
        nav_path = f"/model/{model_id}"
        return nav_path
    
    @app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Output("home-delete-feedback", "children"),
        Input({"type": "home-delete-model", "model_id": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    @logged_callback("delete_model")
    def delete_model(n_clicks_list):
        """Delete model with confirmation."""
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        # Find which button was clicked
        triggered_id = ctx.triggered[0]["prop_id"]
        if ".n_clicks" not in triggered_id:
            raise PreventUpdate
        
        # Extract model_id
        import json
        id_dict = json.loads(triggered_id.split(".")[0])
        model_id = id_dict["model_id"]
        
        # Execute delete command
        command = DeleteModelCommand(model_id=model_id)
        success, message = dashboard_state.dispatcher.execute(command)
        
        if not success:
            # Don't navigate on failure
            feedback = dbc.Alert(
                message,
                color="warning",
                dismissable=True,
                is_open=True,
                style={"marginBottom": "16px"},
            )
            return no_update, feedback
        
        # Force page reload - navigate away and back to force refresh
        # Dash doesn't re-render if we're already on the same path
        return "/?refresh=1", ""
