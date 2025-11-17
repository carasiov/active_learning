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
        Output("home-num-samples-input", "value"),
        Output("home-num-labeled-input", "value"),
        Output("home-seed-input", "value"),
        Output("home-encoder-type-radio", "value"),
        Output("home-hidden-dims-input", "value"),
        Output("home-latent-dim-input", "value"),
        Output("home-recon-loss-radio", "value"),
        Output("home-heteroscedastic-checkbox", "value"),
        Output("home-prior-type-radio", "value"),
        Output("home-num-components-input", "value"),
        Output("home-component-embedding-dim-input", "value"),
        Output("home-component-aware-decoder-checkbox", "value"),
        Input("home-new-model-btn", "n_clicks"),
        Input("home-confirm-create", "n_clicks"),
        Input("home-cancel-create", "n_clicks"),
        State("home-create-modal", "is_open"),
        prevent_initial_call=True,
    )
    @logged_callback("toggle_create_modal")
    def toggle_create_modal(new_clicks, confirm_clicks, cancel_clicks, is_open):
        """Open/close create model modal and reset all inputs."""
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # Default values
        defaults = (
            "",           # model name
            1024,         # num_samples
            128,          # num_labeled
            0,            # seed
            "conv",       # encoder_type
            "256,128,64", # hidden_dims
            2,            # latent_dim
            "bce",        # recon_loss
            False,        # heteroscedastic
            "mixture",    # prior_type
            10,           # num_components
            "",           # component_embedding_dim (empty for auto)
            True,         # component_aware_decoder
        )

        if triggered_id == "home-new-model-btn":
            # Open modal and reset inputs
            return (True,) + defaults
        elif triggered_id == "home-cancel-create":
            # Explicit cancel closes modal
            return (False,) + defaults
        elif triggered_id == "home-confirm-create":
            # Keep modal open; create callback will navigate on success
            # or show feedback on failure
            return (is_open,) + defaults

        raise PreventUpdate

    @app.callback(
        Output("home-hidden-layers-group", "style"),
        Input("home-encoder-type-radio", "value"),
        prevent_initial_call=False,
    )
    def toggle_hidden_layers(encoder_type: str):
        """Show/hide hidden layers input based on encoder type."""
        if encoder_type == "dense":
            return {"marginBottom": "16px", "display": "block"}
        else:
            return {"marginBottom": "16px", "display": "none"}

    @app.callback(
        Output("home-prior-options-group", "style"),
        Input("home-prior-type-radio", "value"),
        prevent_initial_call=False,
    )
    def toggle_prior_options(prior_type: str):
        """Show/hide prior-specific options based on prior type."""
        if prior_type in ["mixture", "vamp", "geometric_mog"]:
            return {"display": "block"}
        else:
            return {"display": "none"}

    @app.callback(
        Output("home-component-aware-decoder-group", "style"),
        Input("home-prior-type-radio", "value"),
        prevent_initial_call=False,
    )
    def toggle_component_aware_option(prior_type: str):
        """Show component-aware decoder option only for mixture/geometric."""
        if prior_type in ["mixture", "geometric_mog"]:
            return {"marginBottom": "16px", "display": "block"}
        else:
            return {"marginBottom": "16px", "display": "none"}
    
    @app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Output("home-create-feedback", "children"),
        Input("home-confirm-create", "n_clicks"),
        State("home-model-name-input", "value"),
        State("home-num-samples-input", "value"),
        State("home-num-labeled-input", "value"),
        State("home-seed-input", "value"),
        State("home-encoder-type-radio", "value"),
        State("home-hidden-dims-input", "value"),
        State("home-latent-dim-input", "value"),
        State("home-recon-loss-radio", "value"),
        State("home-heteroscedastic-checkbox", "value"),
        State("home-prior-type-radio", "value"),
        State("home-num-components-input", "value"),
        State("home-component-embedding-dim-input", "value"),
        State("home-component-aware-decoder-checkbox", "value"),
        prevent_initial_call=True,
    )
    def create_model(
        n_clicks, model_name, num_samples, num_labeled, seed,
        encoder_type, hidden_dims, latent_dim, recon_loss, heteroscedastic,
        prior_type, num_components, component_embedding_dim, component_aware_decoder
    ):
        """Create new model with full architectural configuration and navigate to it."""
        if not n_clicks:
            raise PreventUpdate

        # Parse component_embedding_dim (empty string means None/auto)
        comp_emb_dim = None
        if component_embedding_dim:
            try:
                comp_emb_dim = int(component_embedding_dim)
            except (ValueError, TypeError):
                comp_emb_dim = None

        command = CreateModelCommand(
            # Dataset parameters
            name=model_name if model_name else None,
            num_samples=int(num_samples) if num_samples is not None else 1024,
            num_labeled=int(num_labeled) if num_labeled is not None else 128,
            seed=int(seed) if seed is not None else 0,

            # Architecture parameters
            encoder_type=encoder_type or "conv",
            decoder_type=encoder_type or "conv",  # Match encoder type
            hidden_dims=hidden_dims if hidden_dims else None,
            latent_dim=int(latent_dim) if latent_dim is not None else 2,
            reconstruction_loss=recon_loss or "bce",
            use_heteroscedastic_decoder=bool(heteroscedastic),

            # Prior configuration
            prior_type=prior_type or "mixture",
            num_components=int(num_components) if num_components is not None else 10,
            component_embedding_dim=comp_emb_dim,
            use_component_aware_decoder=bool(component_aware_decoder),
        )

        success, model_id_or_error = dashboard_state.state_manager.dispatcher.execute(command)

        if not success:
            return no_update, html.Div(model_id_or_error, style={"color": "#C10A27"})

        # model_id_or_error is actually the model_id on success
        model_id = model_id_or_error

        # Now load the model
        load_command = LoadModelCommand(model_id=model_id)
        load_success, load_message = dashboard_state.state_manager.dispatcher.execute(load_command)

        if not load_success:
            return no_update, html.Div(f"Created but failed to load: {load_message}", style={"color": "#C10A27"})

        # Navigate to model
        return f"/model/{model_id}", ""

    @app.callback(
        Output("home-unlabeled-preview", "children"),
        Input("home-num-samples-input", "value"),
        Input("home-num-labeled-input", "value"),
        prevent_initial_call=False,
    )
    def update_unlabeled_preview(total_value, labeled_value):
        """Display derived unlabeled sample count in the modal."""
        try:
            total = int(total_value) if total_value is not None else 0
            labeled = int(labeled_value) if labeled_value is not None else 0
        except (TypeError, ValueError):
            return ""

        if total <= 0:
            return "Specify a positive total sample count."
        if labeled < 0:
            return "Labeled samples cannot be negative."
        if labeled > total:
            return "Labeled samples exceed total; adjust the values."

        unlabeled = total - labeled
        return f"Unlabeled samples: {unlabeled:,}"
    
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
        
        # Check if this was an actual click (value changed and > 0)
        triggered_value = ctx.triggered[0]["value"]
        if triggered_value is None or triggered_value == 0:
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
        success, message = dashboard_state.state_manager.dispatcher.execute(command)
        
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
        
        # Check if this was an actual click (value changed and > 0)
        triggered_value = ctx.triggered[0]["value"]
        if triggered_value is None or triggered_value == 0:
            raise PreventUpdate
        
        # Extract model_id
        import json
        id_dict = json.loads(triggered_id.split(".")[0])
        model_id = id_dict["model_id"]
        
        # Execute delete command
        command = DeleteModelCommand(model_id=model_id)
        success, message = dashboard_state.state_manager.dispatcher.execute(command)
        
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
