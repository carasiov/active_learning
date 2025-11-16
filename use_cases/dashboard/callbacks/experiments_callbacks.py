from __future__ import annotations

"""Callbacks for the experiment browser page."""

from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs

from dash import Dash, Input, Output, State, no_update

from use_cases.dashboard.core.experiment_catalog import serialize_run_list
from use_cases.dashboard.pages.experiments import build_run_list, render_run_detail


def register_experiments_callbacks(app: Dash) -> None:
    @app.callback(
        Output("experiments-run-data", "data"),
        Input("experiments-refresh-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def refresh_runs(_n_clicks: Optional[int]) -> List[Dict[str, Any]]:
        return serialize_run_list()

    def _parse_filters(search: Optional[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse URL search params for filters and run selection.

        Returns:
            (model_filter, tag_filter, run_prefill)
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
        Output("experiments-run-selector", "options"),
        Output("experiments-run-selector", "value"),
        Output("experiments-filter-indicator", "children"),
        Output("experiments-model-filter", "value"),
        Output("experiments-tag-filter", "value"),
        Input("experiments-run-data", "data"),
        Input("experiments-url", "search"),
        Input("experiments-model-filter", "value"),
        Input("experiments-tag-filter", "value"),
        State("experiments-run-selector", "value"),
        prevent_initial_call=False,
    )
    def populate_run_selector(
        data: Optional[List[Dict[str, Any]]],
        search: Optional[str],
        model_filter_input: Optional[str],
        tag_filter_input: Optional[str],
        current_value: Optional[str],
    ):
        entries = data or []
        url_model_filter, url_tag_filter, run_prefill = _parse_filters(search)

        # Use URL filters if present, otherwise use dropdown values
        model_filter = url_model_filter or model_filter_input
        tag_filter = url_tag_filter or tag_filter_input

        filtered_entries = _filtered_entries(entries, model_filter, tag_filter)

        options = [_option_from_entry(entry) for entry in filtered_entries]
        valid_values = {opt["value"] for opt in options}

        if run_prefill and run_prefill in valid_values:
            selected = run_prefill
        elif current_value in valid_values:
            selected = current_value
        elif options:
            selected = options[0]["value"]
        else:
            selected = None

        # Build filter indicator
        indicator_parts = []
        if model_filter:
            indicator_parts.append(f"Model: {model_filter}")
        if tag_filter:
            indicator_parts.append(f"Tag: {tag_filter}")

        if indicator_parts:
            count = len(filtered_entries)
            indicator = f"{', '.join(indicator_parts)} ({count} run{'s' if count != 1 else ''})"
        else:
            indicator = ""

        return options, selected, indicator, model_filter, tag_filter

    @app.callback(
        Output("experiments-run-list", "children"),
        Input("experiments-run-data", "data"),
        Input("experiments-run-selector", "value"),
        Input("experiments-model-filter", "value"),
        Input("experiments-tag-filter", "value"),
        prevent_initial_call=False,
    )
    def refresh_run_list(
        data: Optional[List[Dict[str, Any]]],
        selected_value: Optional[str],
        model_filter: Optional[str],
        tag_filter: Optional[str],
    ):
        entries = data or []
        filtered_entries = _filtered_entries(entries, model_filter, tag_filter)
        return build_run_list(filtered_entries, selected_value, model_filter)

    @app.callback(
        Output("experiments-run-detail", "children"),
        Input("experiments-run-selector", "value"),
    )
    def show_run_detail(run_id: Optional[str]):
        if not run_id:
            return render_run_detail(None)
        return render_run_detail(run_id)

    @app.callback(
        Output("experiments-model-filter", "value", allow_duplicate=True),
        Output("experiments-tag-filter", "value", allow_duplicate=True),
        Output("experiments-url", "search", allow_duplicate=True),
        Input("experiments-clear-filters-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_filters(_n_clicks: Optional[int]):
        """Clear all experiment filters and reset URL."""
        return None, None, ""


def _option_from_entry(entry: Dict[str, Any]) -> Dict[str, str]:
    run_id = entry.get("run_id", "run")
    timestamp = entry.get("timestamp")
    label = f"{run_id} — {timestamp}" if timestamp else run_id
    return {"label": label, "value": run_id}
