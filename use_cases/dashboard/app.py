"""Dashboard entry point."""

from __future__ import annotations

import sys
from pathlib import Path

from dash import Dash
import dash_bootstrap_components as dbc

# Ensure repository imports work when running without installation.
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(APP_DIR) in sys.path:
    sys.path = [p for p in sys.path if p != str(APP_DIR)]

from use_cases.dashboard.layouts import build_dashboard_layout  # noqa: E402
from use_cases.dashboard.state import initialize_model_and_data  # noqa: E402
from use_cases.dashboard.callbacks.training_callbacks import register_training_callbacks  # noqa: E402
from use_cases.dashboard.callbacks.visualization_callbacks import register_visualization_callbacks  # noqa: E402
from use_cases.dashboard.callbacks.labeling_callbacks import register_labeling_callbacks  # noqa: E402


def create_app() -> Dash:
    """Create and configure the Dash application."""
    initialize_model_and_data()

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=False,
    )
    app.title = "SSVAE Active Learning Dashboard"
    app.layout = build_dashboard_layout()

    register_training_callbacks(app)
    register_visualization_callbacks(app)
    register_labeling_callbacks(app)
    return app


app = create_app()


if __name__ == "__main__":
    initialize_model_and_data()
    app.run_server(debug=False, host="0.0.0.0", port=8050)
