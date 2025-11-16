from __future__ import annotations

"""Experiment browser page."""

from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.development.base_component import Component

from use_cases.dashboard.core.experiment_catalog import (
    RunDetail,
    RunListEntry,
    FigurePreview,
    load_run_detail,
    serialize_run_list,
    get_available_models,
    get_available_tags,
)


def build_experiments_layout() -> html.Div:
    runs = serialize_run_list()

    # Get available models with run counts
    available_models = get_available_models()
    available_tags = get_available_tags()
    tag_options = [{"label": f"Tag: {t}", "value": t} for t in available_tags]

    return html.Div(
        [
            dcc.Location(id="experiments-url", refresh=False),
            dcc.Store(id="experiments-run-data", data=runs),
            dcc.Store(id="experiments-model-list", data=available_models),

            # Header with logo and red accent bar
            html.Div([
                # Top header with logo
                html.Div(
                    [
                        # infoteam Logo (left)
                        html.Div(
                            [
                                dcc.Link(
                                    html.Img(
                                        src="/assets/infoteam_logo_basic.png",
                                        alt="infoteam software",
                                        style={
                                            "height": "50px",
                                            "width": "auto",
                                            "display": "block",
                                        },
                                    ),
                                    href="/",
                                    style={"textDecoration": "none", "display": "block"},
                                ),
                            ],
                            style={"display": "inline-block", "marginRight": "32px"},
                        ),
                        # Title (right)
                        html.Div(
                            [
                                html.H1("Experiment Browser", style={
                                    "fontSize": "24px",
                                    "fontWeight": "700",
                                    "margin": "0",
                                    "color": "#000000",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                }),
                                html.Div("Review training runs by model", style={
                                    "fontSize": "15px",
                                    "color": "#6F6F6F",
                                    "marginTop": "2px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                }),
                            ],
                            style={
                                "display": "inline-block",
                                "marginLeft": "0",
                                "verticalAlign": "middle",
                            },
                        ),
                    ],
                    style={
                        "padding": "16px 32px",
                        "backgroundColor": "#ffffff",
                        "display": "flex",
                        "alignItems": "center",
                    },
                ),
                # Red accent bar (infoteam brand element)
                html.Div(style={
                    "height": "4px",
                    "backgroundColor": "#C10A27",
                    "width": "100%",
                }),
            ]),

            # Filter indicator bar
            html.Div(
                [
                    html.Div(
                        id="experiments-filter-indicator",
                        style={
                            "fontSize": "13px",
                            "color": "#45717A",
                            "fontWeight": "600",
                        },
                    ),
                ],
                style={
                    "padding": "12px 48px",
                    "backgroundColor": "#ffffff",
                    "borderBottom": "1px solid #E5E5E5",
                },
            ),
            html.Div(
                [
                    # Three-panel layout: Models | Runs | Detail
                    html.Div(
                        [
                            # LEFT PANEL: Model List
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                "Models",
                                                style={
                                                    "fontSize": "17px",
                                                    "fontWeight": "700",
                                                    "color": "#000000",
                                                    "marginBottom": "12px",
                                                },
                                            ),
                                            dbc.Button(
                                                "Refresh",
                                                id="experiments-refresh-btn",
                                                color="secondary",
                                                outline=True,
                                                size="sm",
                                                style={"fontWeight": "600", "fontSize": "12px"},
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "justifyContent": "space-between",
                                            "alignItems": "center",
                                            "marginBottom": "16px",
                                        },
                                    ),
                                    html.Div(
                                        id="experiments-model-list-ui",
                                        style={
                                            "display": "flex",
                                            "flexDirection": "column",
                                            "gap": "8px",
                                        },
                                    ),
                                ],
                                style={
                                    "flex": "0 0 22%",
                                    "minWidth": "200px",
                                    "maxHeight": "calc(100vh - 180px)",
                                    "overflowY": "auto",
                                    "paddingRight": "16px",
                                },
                            ),

                            # MIDDLE PANEL: Run List
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                "Training Runs",
                                                style={
                                                    "fontSize": "17px",
                                                    "fontWeight": "700",
                                                    "color": "#000000",
                                                },
                                            ),
                                            # Optional tag filter
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Filter by tag:",
                                                        style={"fontSize": "12px", "color": "#6F6F6F", "marginRight": "8px"},
                                                    ),
                                                    dcc.Dropdown(
                                                        id="experiments-tag-filter",
                                                        options=tag_options,
                                                        value=None,
                                                        clearable=True,
                                                        placeholder="All tags",
                                                        style={"fontSize": "13px", "minWidth": "150px"},
                                                    ),
                                                ],
                                                style={
                                                    "display": "flex",
                                                    "alignItems": "center",
                                                    "gap": "8px",
                                                },
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "justifyContent": "space-between",
                                            "alignItems": "center",
                                            "marginBottom": "16px",
                                            "paddingBottom": "12px",
                                            "borderBottom": "1px solid #E5E5E5",
                                        },
                                    ),
                                    html.Div(
                                        id="experiments-run-list",
                                        style={
                                            "display": "flex",
                                            "flexDirection": "column",
                                            "gap": "8px",
                                        },
                                    ),
                                ],
                                style={
                                    "flex": "0 0 32%",
                                    "minWidth": "280px",
                                    "maxHeight": "calc(100vh - 180px)",
                                    "overflowY": "auto",
                                    "paddingLeft": "16px",
                                    "paddingRight": "16px",
                                    "borderLeft": "1px solid #E5E5E5",
                                },
                            ),

                            # RIGHT PANEL: Run Detail
                            html.Div(
                                id="experiments-run-detail",
                                style={
                                    "flex": "1",
                                    "maxHeight": "calc(100vh - 180px)",
                                    "overflowY": "auto",
                                    "paddingLeft": "16px",
                                    "borderLeft": "1px solid #E5E5E5",
                                },
                            ),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "flex-start",
                            "padding": "24px 48px 32px 48px",
                            "gap": "0",
                        },
                    ),
                ],
                style={"backgroundColor": "#fafafa", "minHeight": "calc(100vh - 120px)"},
            ),
        ],
        style={"backgroundColor": "#fafafa", "minHeight": "100vh"},
    )


def build_model_list(models: List[str], runs_data: List[Dict[str, Any]], selected_model: Optional[str]) -> List[Component]:
    """Build list of model cards with run counts."""
    if not models:
        return [
            html.Div(
                "No models with training runs found.",
                style={
                    "padding": "16px",
                    "backgroundColor": "#ffffff",
                    "borderRadius": "8px",
                    "border": "1px dashed #C6C6C6",
                    "color": "#6F6F6F",
                    "fontStyle": "italic",
                    "fontSize": "13px",
                },
            )
        ]

    cards: List[Component] = []
    for model_id in models:
        # Count runs for this model
        run_count = sum(1 for run in runs_data if run.get("model_id") == model_id)

        # Find latest run timestamp
        model_runs = [run for run in runs_data if run.get("model_id") == model_id]
        latest_timestamp = model_runs[0].get("timestamp", "") if model_runs else ""

        is_selected = (model_id == selected_model)

        card = html.Div(
            [
                html.Div(
                    model_id,
                    style={
                        "fontWeight": "600",
                        "fontSize": "14px",
                        "color": "#000000",
                        "marginBottom": "4px",
                    },
                ),
                html.Div(
                    f"{run_count} run{'s' if run_count != 1 else ''}",
                    style={
                        "fontSize": "12px",
                        "color": "#6F6F6F",
                    },
                ),
                html.Div(
                    latest_timestamp[:10] if latest_timestamp else "",
                    style={
                        "fontSize": "11px",
                        "color": "#999",
                        "marginTop": "2px",
                    },
                ) if latest_timestamp else None,
            ],
            style={
                "padding": "12px",
                "backgroundColor": "#FFE5EA" if is_selected else "#ffffff",
                "border": f"2px solid {'#C10A27' if is_selected else '#E5E5E5'}",
                "borderRadius": "8px",
                "cursor": "pointer",
                "transition": "all 0.2s",
            },
        )

        # Wrap in link with URL parameter
        href = f"/experiments?model={model_id}"
        cards.append(dcc.Link(card, href=href, style={"textDecoration": "none", "display": "block"}))

    return cards


def build_run_list(entries: List[Dict[str, Any]], selected: Optional[str], model_filter: Optional[str] = None) -> List[Component]:
    cards: List[Component] = []
    for entry_dict in entries:
        entry = _entry_from_dict(entry_dict)
        cards.append(_run_card(entry, selected == entry.run_id, model_filter))
    if not cards:
        cards.append(
            html.Div(
                "No experiment runs recorded yet.",
                style={
                    "padding": "24px",
                    "backgroundColor": "#ffffff",
                    "borderRadius": "8px",
                    "border": "1px dashed #C6C6C6",
                    "color": "#6F6F6F",
                    "fontStyle": "italic",
                },
            )
        )
    return cards


def render_run_detail(run_id: Optional[str]) -> html.Div:
    if not run_id:
        return _detail_placeholder("Select an experiment run to inspect its outputs.")

    detail = load_run_detail(run_id)
    if detail is None:
        return _detail_placeholder("Run artifacts not found on disk.")

    return _detail_view(detail)


def _run_option(entry: Dict[str, Any]) -> Dict[str, str]:
    run_id = entry["run_id"]
    timestamp = entry.get("timestamp", "")
    label = f"{run_id} — {timestamp}" if timestamp else run_id
    return {"label": label, "value": run_id}


def _entry_from_dict(data: Dict[str, Any]) -> RunListEntry:
    return RunListEntry(
        run_id=data["run_id"],
        experiment_name=data.get("experiment_name", data["run_id"]),
        timestamp=data.get("timestamp", ""),
        architecture_code=data.get("architecture_code"),
        status=data.get("status", "unknown"),
        path=Path(data.get("path", "")) if data.get("path") else Path("."),
        model_id=data.get("model_id"),
        tags=data.get("tags", []),
        metrics=data.get("metrics", {}),
        summary_path=Path(data["summary_path"]) if data.get("summary_path") else None,
    )


def _run_card(entry: RunListEntry, is_selected: bool, model_filter: Optional[str]) -> Component:
    badge_color = "success" if entry.status == "complete" else "warning"
    metrics = _format_metrics(entry.metrics)

    query_params = {"run": entry.run_id}
    if model_filter:
        query_params["model"] = model_filter
    href = f"/experiments?{urlencode(query_params)}"

    card = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(entry.experiment_name, style={"fontWeight": "600", "fontSize": "16px"}),
                            html.Div(
                                entry.timestamp,
                                style={"fontSize": "12px", "color": "#6F6F6F"},
                            ),
                        ],
                    ),
                    html.Div(
                        dbc.Badge(entry.status.title(), color=badge_color, pill=True),
                        style={"textAlign": "right"},
                    ),
                ],
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
            ),
            html.Div(metrics, style={"marginTop": "12px", "display": "flex", "flexWrap": "wrap", "gap": "6px"}),
            html.Div(
                _format_tags(entry),
                style={"marginTop": "12px", "display": "flex", "flexWrap": "wrap", "gap": "4px"},
            ),
        ],
        style={
            "backgroundColor": "#ffffff" if not is_selected else "#FFE5EA",
            "border": f"2px solid {'#C10A27' if is_selected else '#E5E5E5'}",
            "borderRadius": "10px",
            "padding": "18px",
            "marginBottom": "12px",
            "cursor": "pointer",
        },
    )

    return dcc.Link(card, href=href, style={"textDecoration": "none", "display": "block"})


def _format_metrics(metrics: Dict[str, Any]) -> List[dbc.Badge]:
    badges: List[dbc.Badge] = []
    for idx, (key, value) in enumerate(metrics.items()):
        if idx >= 4:
            break
        label = key.replace("_", " ").title()
        if isinstance(value, float):
            text = f"{label}: {value:.4f}"
        else:
            text = f"{label}: {value}"
        badges.append(dbc.Badge(text, color="light", text_color="black"))
    return badges


def _format_tags(entry: RunListEntry) -> List[dbc.Badge]:
    tags: List[dbc.Badge] = []
    if entry.model_id:
        tags.append(dbc.Badge(f"Model {entry.model_id}", color="dark", pill=True))
    if entry.architecture_code:
        tags.append(dbc.Badge(entry.architecture_code, color="secondary", pill=True))
    for tag in entry.tags:
        tags.append(dbc.Badge(tag, color="light", text_color="black"))
    return tags


def _detail_placeholder(message: str) -> html.Div:
    return html.Div(
        message,
        style={
            "padding": "32px",
            "backgroundColor": "#ffffff",
            "borderRadius": "12px",
            "border": "1px dashed #C6C6C6",
            "color": "#6F6F6F",
            "fontStyle": "italic",
        },
    )


def _detail_view(detail: RunDetail) -> html.Div:
    header = detail.entry
    summary_sections = _render_summary(detail.summary)
    figures = _render_figures(detail.figures)
    artifacts = _render_artifacts(detail.artifacts)

    return html.Div(
        [
            html.Div(
                [
                    html.H2(header.run_id, style={"fontSize": "22px", "fontWeight": "700", "margin": "0"}),
                    html.Div(
                        header.timestamp,
                        style={"fontSize": "13px", "color": "#6F6F6F", "marginTop": "4px"},
                    ),
                    html.Div(
                        _format_tags(header),
                        style={"marginTop": "12px", "display": "flex", "gap": "6px", "flexWrap": "wrap"},
                    ),
                ],
                style={"marginBottom": "18px"},
            ),
            html.H4("Summary", style={"fontSize": "18px", "fontWeight": "600"}),
            *summary_sections,
            html.H4("Configuration", style={"fontSize": "18px", "fontWeight": "600", "marginTop": "24px"}),
            _render_config(detail.config_text),
            html.H4("Figures", style={"fontSize": "18px", "fontWeight": "600", "marginTop": "24px"}),
            figures,
            html.H4("Artifacts", style={"fontSize": "18px", "fontWeight": "600", "marginTop": "24px"}),
            artifacts,
        ],
        style={"backgroundColor": "#ffffff", "padding": "24px", "borderRadius": "12px"},
    )


def _render_summary(summary: Dict[str, Any]) -> List[html.Div]:
    sections: List[html.Div] = []
    for section_name in ["training", "classification", "mixture", "tau_classifier", "clustering"]:
        section = summary.get(section_name)
        if not isinstance(section, dict) or not section:
            continue
        rows = []
        for key, value in section.items():
            label = key.replace("_", " ").title()
            if isinstance(value, float):
                text = f"{value:.4f}"
            else:
                text = str(value)
            rows.append(html.Tr([html.Td(label), html.Td(text)]))
        sections.append(
            html.Div(
                [
                    html.H5(section_name.replace("_", " ").title(), style={"fontSize": "15px", "fontWeight": "600"}),
                    dbc.Table(html.Tbody(rows), bordered=True, size="sm"),
                ],
                style={"marginBottom": "16px"},
            )
        )
    if not sections:
        sections.append(
            html.Div(
                "Summary metrics unavailable.",
                style={"color": "#6F6F6F", "fontStyle": "italic"},
            )
        )
    return sections


def _render_config(config_text: Optional[str]) -> Any:
    if not config_text:
        return html.Div("Config file missing.", style={"color": "#6F6F6F", "fontStyle": "italic"})
    return html.Pre(
        config_text,
        style={
            "backgroundColor": "#1E1E1E",
            "color": "#F8F8F2",
            "padding": "16px",
            "borderRadius": "8px",
            "overflowX": "auto",
            "fontSize": "13px",
        },
    )


def _render_figures(figures: List[FigurePreview]) -> Any:
    if not figures:
        return html.Div("No figures generated.", style={"color": "#6F6F6F", "fontStyle": "italic"})

    groups: Dict[str, List[FigurePreview]] = {}
    for fig in figures:
        groups.setdefault(fig.category, []).append(fig)

    panels: List[dbc.AccordionItem] = []
    for category, items in groups.items():
        body = []
        for figure in items:
            img = html.Img(
                src=figure.data_url,
                style={"maxWidth": "100%", "borderRadius": "8px", "marginBottom": "12px"},
            ) if figure.data_url else html.Div(
                f"Image exceeds inline size limit ({figure.relative_path})",
                style={"color": "#6F6F6F", "fontStyle": "italic", "marginBottom": "12px"},
            )
            body.append(
                html.Div(
                    [
                        html.Div(figure.name, style={"fontWeight": "600", "marginBottom": "6px"}),
                        img,
                        html.Div(figure.relative_path, style={"fontSize": "12px", "color": "#6F6F6F"}),
                    ],
                    style={"marginBottom": "16px"},
                )
            )
        panels.append(
            dbc.AccordionItem(
                body,
                title=category.replace("_", " ").title(),
            )
        )

    return dbc.Accordion(panels, always_open=True)


def _render_artifacts(artifacts: Dict[str, List[str]]) -> Any:
    if not artifacts:
        return html.Div("No saved artifacts.", style={"color": "#6F6F6F", "fontStyle": "italic"})
    rows = []
    for category, files in artifacts.items():
        rows.append(
            html.Tr(
                [
                    html.Td(category.replace("_", " ").title()),
                    html.Td(
                        html.Ul([html.Li(file) for file in files], style={"paddingLeft": "18px", "margin": "0"})
                    ),
                ]
            )
        )
    return dbc.Table(html.Tbody(rows), bordered=True, size="sm")