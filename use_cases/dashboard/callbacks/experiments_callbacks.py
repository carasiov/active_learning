from __future__ import annotations

"""Callbacks for the experiment browser page."""

from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs

from dash import Dash, Input, Output, State, no_update

from use_cases.dashboard.core.experiment_catalog import serialize_run_list, get_available_models
from use_cases.dashboard.pages.experiments import build_model_list, build_run_list, render_run_detail


def register_experiments_callbacks(app: Dash) -> None:
    @app.callback(
        Output("experiments-run-data", "data"),
        Output("experiments-model-list", "data"),
        Input("experiments-refresh-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def refresh_data(_n_clicks: Optional[int]):
        """Refresh both runs and model list."""
        runs = serialize_run_list()
        models = get_available_models()
        return runs, models

    def _parse_url(search: Optional[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse URL search params for model, tag, and run selection.

        Returns:
            (model_id, tag_filter, run_id)
        """
        if not search:
            return None, None, None
        query = parse_qs(search.lstrip("?"))
        return (
            query.get("model", [None])[0],
            query.get("tag", [None])[0],
            query.get("run", [None])[0]
        )

    def _filtered_entries(
        entries: List[Dict[str, Any]],
        model_filter: Optional[str],
        tag_filter: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Filter experiment entries by model ID and/or tag."""
        filtered = entries

        if model_filter:
            filtered = [entry for entry in filtered if entry.get("model_id") == model_filter]

        if tag_filter:
            filtered = [entry for entry in filtered if tag_filter in entry.get("tags", [])]

        return filtered

    @app.callback(
        Output("experiments-model-list-ui", "children"),
        Input("experiments-run-data", "data"),
        Input("experiments-model-list", "data"),
        Input("experiments-url", "search"),
        prevent_initial_call=False,
    )
    def refresh_model_list(
        runs_data: Optional[List[Dict[str, Any]]],
        models: Optional[List[str]],
        search: Optional[str],
    ):
        """Render model list with selection state from URL."""
        selected_model, _, _ = _parse_url(search)
        return build_model_list(models or [], runs_data or [], selected_model)

    @app.callback(
        Output("experiments-filter-indicator", "children"),
        Input("experiments-url", "search"),
        Input("experiments-tag-filter", "value"),
        State("experiments-run-data", "data"),
        prevent_initial_call=False,
    )
    def update_filter_indicator(
        search: Optional[str],
        tag_filter_input: Optional[str],
        runs_data: Optional[List[Dict[str, Any]]],
    ):
        """Update filter indicator to show active filters and run count."""
        model_filter, url_tag_filter, _ = _parse_url(search)
        tag_filter = url_tag_filter or tag_filter_input

        filtered = _filtered_entries(runs_data or [], model_filter, tag_filter)

        indicator_parts = []
        if model_filter:
            indicator_parts.append(f"Model: {model_filter}")
        if tag_filter:
            indicator_parts.append(f"Tag: {tag_filter}")

        if indicator_parts:
            count = len(filtered)
            return f"{', '.join(indicator_parts)} ({count} run{'s' if count != 1 else ''})"
        return ""

    @app.callback(
        Output("experiments-run-list", "children"),
        Output("experiments-run-detail", "children"),
        Input("experiments-run-data", "data"),
        Input("experiments-url", "search"),
        Input("experiments-tag-filter", "value"),
        prevent_initial_call=False,
    )
    def refresh_runs_and_detail(
        data: Optional[List[Dict[str, Any]]],
        search: Optional[str],
        tag_filter_input: Optional[str],
    ):
        """Refresh run list and detail based on URL and tag filter."""
        model_filter, url_tag_filter, run_id = _parse_url(search)
        tag_filter = url_tag_filter or tag_filter_input

        entries = data or []
        filtered_entries = _filtered_entries(entries, model_filter, tag_filter)

        # Select first run if no run specified in URL
        selected_run_id = run_id
        if not selected_run_id and filtered_entries:
            selected_run_id = filtered_entries[0].get("run_id")

        run_list = build_run_list(filtered_entries, selected_run_id, model_filter)
        run_detail = render_run_detail(selected_run_id)

        return run_list, run_detail


def _option_from_entry(entry: Dict[str, Any]) -> Dict[str, str]:
    run_id = entry.get("run_id", "run")
    timestamp = entry.get("timestamp")
    label = f"{run_id} — {timestamp}" if timestamp else run_id
    return {"label": label, "value": run_id}
