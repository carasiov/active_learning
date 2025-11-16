"""Legacy configuration callback module.

The dynamic configuration UI now lives in ``use_cases.dashboard.pages.training`` and
registers its callbacks via ``register_config_page_callbacks``. We keep this module
to preserve the public import used in ``app.py``; the registration is currently a
no-op but provides a convenient place for future cross-page configuration hooks.
"""

from __future__ import annotations

from dash import Dash


def register_config_callbacks(app: Dash) -> None:  # pragma: no cover - intentional no-op
    """Placeholder for additional configuration callbacks.

    Configuration-specific interactions are registered in
    ``register_config_page_callbacks``. This function exists so that the
    application wiring in ``app.py`` remains stable while the configuration UI
    evolves. When new global config callbacks are needed, add them here.
    """
    return None
