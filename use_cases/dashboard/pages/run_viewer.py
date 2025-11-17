"""Run Viewer page - Display experiment REPORT.md for a completed training run."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from dash import dcc, html
import dash_bootstrap_components as dbc

# Ensure repository imports work
ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from use_cases.dashboard.core import state as dashboard_state
from use_cases.dashboard.core.model_runs import load_run_records
from use_cases.dashboard.core.state_models import RunRecord


def build_run_viewer_layout(model_id: str, run_id: str) -> html.Div:
    """Build the Run Viewer page layout.

    Args:
        model_id: ID of the model
        run_id: ID of the specific run to display

    Returns:
        Dash layout for the run viewer page
    """
    # Load run records for this model
    run_records = load_run_records(model_id)

    # Find the specific run
    run_record: Optional[RunRecord] = None
    for record in run_records:
        if record.run_id == run_id:
            run_record = record
            break

    if run_record is None:
        return html.Div([
            html.H3("Run Not Found", style={"textAlign": "center", "marginTop": "100px"}),
            html.P(f"Run {run_id} not found for model {model_id}", style={"textAlign": "center"}),
            html.A("Go Back", href=f"/model/{model_id}/training-hub", style={"display": "block", "textAlign": "center"})
        ])

    # Read REPORT.md content
    report_content = ""
    if run_record.report_path:
        report_path = Path(run_record.report_path)
        if report_path.exists():
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    report_content = f.read()

                # Note: Image paths in REPORT.md are relative (e.g., figures/core/loss_comparison.png)
                # For now, we display the markdown as-is. Images won't render in the web UI.
                # TODO: Future enhancement - serve figures directory as static assets or convert to base64

            except Exception as e:
                report_content = f"**Error reading report:** {e}"
        else:
            report_content = f"**Report file not found:** {run_record.report_path}"
    else:
        report_content = "**No report path available for this run.**"

    # Get metrics summary
    metrics = run_record.metrics
    val_loss = metrics.get("val_loss", metrics.get("loss", 0.0))

    return html.Div(
        [
            # Header with logo and red accent bar
            html.Div([
                html.Div(
                    [
                        html.Div(
                            [
                                dcc.Link(
                                    html.Img(
                                        src="/assets/infoteam_logo_basic.png",
                                        alt="infoteam software",
                                        style={"height": "50px", "width": "auto", "display": "block"},
                                    ),
                                    href="/",
                                    style={"textDecoration": "none", "display": "block"},
                                ),
                            ],
                            style={"display": "inline-block", "marginRight": "32px"},
                        ),
                        html.Div(
                            [
                                html.H1(f"Run Report: {run_id}", style={
                                    "fontSize": "24px",
                                    "fontWeight": "700",
                                    "margin": "0",
                                    "color": "#000000",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                }),
                                html.Div(f"{run_record.timestamp} • {run_record.epochs_completed} epochs • Loss: {val_loss:.4f}", style={
                                    "fontSize": "15px",
                                    "color": "#6F6F6F",
                                    "marginTop": "2px",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                }),
                            ],
                            style={"display": "inline-block", "verticalAlign": "middle"},
                        ),
                        html.Div(
                            dcc.Link(
                                "← Back to Training Hub",
                                href=f"/model/{model_id}/training-hub",
                                style={
                                    "fontSize": "14px",
                                    "color": "#45717A",
                                    "textDecoration": "none",
                                    "fontWeight": "600",
                                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                                },
                            ),
                            style={"marginLeft": "auto", "display": "inline-block"},
                        ),
                    ],
                    style={
                        "padding": "16px 32px",
                        "backgroundColor": "#ffffff",
                        "display": "flex",
                        "alignItems": "center",
                    },
                ),
                html.Div(style={
                    "height": "4px",
                    "backgroundColor": "#C10A27",
                    "width": "100%",
                }),
            ]),

            # Run metadata summary
            html.Div(
                [
                    html.Div(
                        [
                            html.Div([
                                html.Span("Epochs:", style={"fontWeight": "600", "marginRight": "8px"}),
                                html.Span(f"{run_record.start_epoch}→{run_record.end_epoch}"),
                            ], style={"marginRight": "24px"}),
                            html.Div([
                                html.Span("Labeled Samples:", style={"fontWeight": "600", "marginRight": "8px"}),
                                html.Span(f"{run_record.labeled_samples}/{run_record.total_samples}"),
                            ], style={"marginRight": "24px"}),
                            html.Div([
                                html.Span("Training Time:", style={"fontWeight": "600", "marginRight": "8px"}),
                                html.Span(f"{run_record.train_time_sec:.1f}s"),
                            ], style={"marginRight": "24px"}),
                            html.Div([
                                html.Span("Val Loss:", style={"fontWeight": "600", "marginRight": "8px"}),
                                html.Span(f"{val_loss:.4f}"),
                            ]),
                        ],
                        style={
                            "display": "flex",
                            "fontSize": "14px",
                            "fontFamily": "'Open Sans', Verdana, sans-serif",
                            "color": "#4A4A4A",
                        },
                    ),
                ],
                style={
                    "padding": "16px 32px",
                    "backgroundColor": "#f5f5f5",
                    "borderBottom": "1px solid #C6C6C6",
                },
            ),

            # Info banner
            html.Div(
                [
                    html.Div("ℹ️", style={"marginRight": "12px", "fontSize": "18px"}),
                    html.Div(
                        [
                            html.Strong("Note: "),
                            html.Span("Embedded images are not yet supported in the web viewer. For full visualizations, view the report file directly at: "),
                            html.Code(str(run_record.report_path) if run_record.report_path else "N/A", style={
                                "backgroundColor": "#f5f5f5",
                                "padding": "2px 6px",
                                "borderRadius": "3px",
                                "fontSize": "12px",
                            }),
                        ],
                        style={"flex": "1"},
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "flex-start",
                    "padding": "16px 24px",
                    "backgroundColor": "#E8F4F8",
                    "border": "1px solid #B3D9E8",
                    "borderRadius": "6px",
                    "margin": "32px 32px 16px 32px",
                    "fontSize": "13px",
                    "fontFamily": "'Open Sans', Verdana, sans-serif",
                    "color": "#2C5F77",
                },
            ),

            # Report content
            html.Div(
                [
                    dcc.Markdown(
                        report_content,
                        style={
                            "fontFamily": "'Open Sans', Verdana, sans-serif",
                            "fontSize": "14px",
                            "lineHeight": "1.6",
                            "color": "#4A4A4A",
                        },
                    ),
                ],
                style={
                    "padding": "32px",
                    "backgroundColor": "#ffffff",
                    "margin": "16px 32px 32px 32px",
                    "borderRadius": "8px",
                    "border": "1px solid #C6C6C6",
                },
            ),
        ],
        style={
            "fontFamily": "'Open Sans', Verdana, sans-serif",
            "backgroundColor": "#fafafa",
            "minHeight": "100vh",
        },
    )
