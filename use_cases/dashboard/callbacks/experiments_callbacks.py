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

    def _parse_filters(search: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        if not search:
            return None, None
        query = parse_qs(search.lstrip("?"))
        return query.get("model", [None])[0], query.get("run", [None])[0]

    def _filtered_entries(entries: List[Dict[str, Any]], model_filter: Optional[str]) -> List[Dict[str, Any]]:
        if model_filter:
            return [entry for entry in entries if entry.get("model_id") == model_filter]
        return entries

    @app.callback(
        Output("experiments-run-selector", "options"),
        Output("experiments-run-selector", "value"),
        Output("experiments-filter-indicator", "children"),
        Input("experiments-run-data", "data"),
        Input("experiments-url", "search"),
        State("experiments-run-selector", "value"),
        prevent_initial_call=False,
    )
    def populate_run_selector(
        data: Optional[List[Dict[str, Any]]],
        search: Optional[str],
        current_value: Optional[str],
    ):
        entries = data or []
        model_filter, run_prefill = _parse_filters(search)
        filtered_entries = _filtered_entries(entries, model_filter)

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

        indicator = ""
        if model_filter:
            count = len(filtered_entries)
            indicator = f"Filtered to model {model_filter} ({count} run{'s' if count != 1 else ''})"

        return options, selected, indicator

    @app.callback(
        Output("experiments-run-list", "children"),
        Input("experiments-run-data", "data"),
        Input("experiments-run-selector", "value"),
        Input("experiments-url", "search"),
        prevent_initial_call=False,
    )
    def refresh_run_list(
        data: Optional[List[Dict[str, Any]]],
        selected_value: Optional[str],
        search: Optional[str],
    ):
        entries = data or []
        model_filter, _ = _parse_filters(search)
        filtered_entries = _filtered_entries(entries, model_filter)
        return build_run_list(filtered_entries, selected_value)

    @app.callback(
        Output("experiments-run-detail", "children"),
        Input("experiments-run-selector", "value"),
    )
    def show_run_detail(run_id: Optional[str]):
        if not run_id:
            return render_run_detail(None)
        return render_run_detail(run_id)


def _option_from_entry(entry: Dict[str, Any]) -> Dict[str, str]:
    run_id = entry.get("run_id", "run")
    timestamp = entry.get("timestamp")
    label = f"{run_id} — {timestamp}" if timestamp else run_id
    return {"label": label, "value": run_id}
