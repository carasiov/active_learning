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
        Output("home-encoder-type", "value"),
        Output("home-latent-dim", "value"),
        Output("home-hidden-dims", "value"),
        Output("home-prior-type", "value"),
        Output("home-num-components", "value"),
        Output("home-component-embedding-dim", "value"),
        Output("home-use-component-aware-decoder", "value"),
        Input("home-new-model-btn", "n_clicks"),
        Input("home-confirm-create", "n_clicks"),
        Input("home-cancel-create", "n_clicks"),
        State("home-create-modal", "is_open"),
        prevent_initial_call=True,
    )
    @logged_callback("toggle_create_modal")
    def toggle_create_modal(new_clicks, confirm_clicks, cancel_clicks, is_open):
        """Open/close create model modal and reset inputs."""
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # Default values
        defaults = (
            "",           # name
            1024,         # num_samples
            128,          # num_labeled
            0,            # seed
            "conv",       # encoder_type
            2,            # latent_dim
            "256,128,64", # hidden_dims
            "standard",   # prior_type
            10,           # num_components
            "",           # component_embedding_dim (blank = use latent_dim)
            [True],       # use_component_aware_decoder
        )

        if triggered_id == "home-new-model-btn":
            # Open modal and reset inputs to defaults
            return (True,) + defaults
        elif triggered_id == "home-cancel-create":
            # Explicit cancel closes modal
            return (False,) + defaults
        elif triggered_id == "home-confirm-create":
            # Keep modal open; create callback will navigate on success
            return (is_open,) + defaults

        raise PreventUpdate
    
    @app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Output("home-create-feedback", "children"),
        Input("home-confirm-create", "n_clicks"),
        State("home-model-name-input", "value"),
        State("home-num-samples-input", "value"),
        State("home-num-labeled-input", "value"),
        State("home-seed-input", "value"),
        State("home-encoder-type", "value"),
        State("home-latent-dim", "value"),
        State("home-hidden-dims", "value"),
        State("home-prior-type", "value"),
        State("home-num-components", "value"),
        State("home-component-embedding-dim", "value"),
        State("home-use-component-aware-decoder", "value"),
        prevent_initial_call=True,
    )
    def create_model(
        n_clicks, model_name, num_samples, num_labeled, seed,
        encoder_type, latent_dim, hidden_dims,
        prior_type, num_components, component_embedding_dim,
        use_component_aware_decoder
    ):
        """Create new model with architecture configuration and navigate to it."""
        if not n_clicks:
            raise PreventUpdate

        # Parse hidden_dims
        try:
            if hidden_dims:
                hidden_dims_tuple = tuple(int(x.strip()) for x in hidden_dims.split(",") if x.strip())
            else:
                hidden_dims_tuple = (256, 128, 64)
        except ValueError:
            return no_update, html.Div("Invalid hidden_dims format. Use comma-separated integers (e.g., 256,128,64)", style={"color": "#C10A27"})

        # Parse component_embedding_dim (empty string means None)
        comp_emb_dim = None
        if component_embedding_dim not in (None, ""):
            try:
                comp_emb_dim = int(component_embedding_dim)
            except ValueError:
                return no_update, html.Div("Component embedding dim must be an integer or blank", style={"color": "#C10A27"})

        command = CreateModelCommand(
            name=model_name if model_name else None,
            num_samples=int(num_samples) if num_samples is not None else 1024,
            num_labeled=int(num_labeled) if num_labeled is not None else 128,
            seed=int(seed) if seed is not None else 0,
            # Architecture configuration
            encoder_type=encoder_type or "conv",
            latent_dim=int(latent_dim) if latent_dim is not None else 2,
            hidden_dims=hidden_dims_tuple,
            prior_type=prior_type or "standard",
            num_components=int(num_components) if num_components is not None else 10,
            component_embedding_dim=comp_emb_dim,
            use_component_aware_decoder=bool(use_component_aware_decoder),
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
        Output("home-hidden-dims-container", "style"),
        Input("home-encoder-type", "value"),
        prevent_initial_call=False,
    )
    def toggle_hidden_dims(encoder_type):
        """Show/hide hidden_dims input based on encoder type (Dense only)."""
        if encoder_type == "dense":
            return {"marginBottom": "16px", "display": "block"}
        else:
            return {"marginBottom": "16px", "display": "none"}

    @app.callback(
        Output("home-mixture-options", "style"),
        Output("home-component-aware-container", "style"),
        Output("home-prior-help", "children"),
        Input("home-prior-type", "value"),
        prevent_initial_call=False,
    )
    def toggle_mixture_options(prior_type):
        """Show/hide mixture-specific options based on prior type."""
        help_texts = {
            "standard": "Simple Gaussian prior N(0,I) - no mixture structure",
            "mixture": "Mixture of Gaussians with component-aware decoder. Enables τ-classifier.",
            "vamp": "Variational Mixture of Posteriors - learned pseudo-inputs provide spatial separation",
            "geometric_mog": "Fixed geometric mixture (circle/grid) - diagnostic tool only (WARNING: induces topology)",
        }

        help_text = help_texts.get(prior_type, "")

        mixture_based = prior_type in ["mixture", "vamp", "geometric_mog"]

        if mixture_based:
            return (
                {"display": "flex", "marginBottom": "16px"},  # mixture_options
                {"display": "block", "marginBottom": "16px"},  # component_aware_container
                help_text,  # help text
            )
        else:
            return (
                {"display": "none", "marginBottom": "16px"},
                {"display": "none", "marginBottom": "16px"},
                help_text,
            )

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
