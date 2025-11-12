"""Visualization registry for experiments.

Handles ComponentResult for explicit status tracking of plot generation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from ..metrics.status import ComponentResult

logger = logging.getLogger("experiment")

PlotResult = Dict[str, Any]
Plotter = Callable[["VisualizationContext"], Union[ComponentResult, Optional[PlotResult]]]


@dataclass(slots=True)
class VisualizationContext:
    model: Any
    config: Any
    history: Dict[str, List[float]]
    x_train: np.ndarray
    y_true: np.ndarray
    figures_dir: Path


_PLOTTERS: List[Plotter] = []


def register_plotter(func: Plotter) -> Plotter:
    _PLOTTERS.append(func)
    return func


def render_all_plots(context: VisualizationContext) -> PlotResult:
    """Render all registered plots and aggregate their results.

    Supports both ComponentResult  and legacy dict/None returns.

    Returns:
        Dictionary containing plot metadata and status information.
    """
    context.figures_dir.mkdir(parents=True, exist_ok=True)
    aggregate: PlotResult = {}
    status_summary = {}

    for plotter in _PLOTTERS:
        plotter_name = plotter.__name__.replace('_plotter', '')

        try:
            result = plotter(context)

            # Handle ComponentResult 
            if isinstance(result, ComponentResult):
                if result.is_success:
                    logger.debug(f"✓ {plotter_name}: plot generated")
                    if result.data:
                        aggregate.update(result.data)
                    status_summary[plotter_name] = result.to_dict()

                elif result.is_disabled:
                    logger.debug(f"○ {plotter_name}: disabled ({result.reason})")
                    status_summary[plotter_name] = result.to_dict()

                elif result.is_skipped:
                    logger.debug(f"⊘ {plotter_name}: skipped ({result.reason})")
                    status_summary[plotter_name] = result.to_dict()

                else:  # FAILED
                    logger.warning(f"✗ {plotter_name}: failed ({result.reason})")
                    if result.error:
                        logger.debug(f"  Error details: {result.error}", exc_info=result.error)
                    status_summary[plotter_name] = result.to_dict()

            # Handle legacy dict/None (backward compatibility)
            elif isinstance(result, dict):
                logger.debug(f"✓ {plotter_name}: plot generated (legacy)")
                aggregate.update(result)
                status_summary[plotter_name] = {"status": "success", "legacy": True}

            elif result is None:
                # Treat None as disabled (legacy behavior)
                logger.debug(f"○ {plotter_name}: disabled (legacy None return)")
                status_summary[plotter_name] = {"status": "disabled", "legacy": True}

            else:
                logger.warning(f"✗ {plotter_name}: unexpected return type {type(result)}")
                status_summary[plotter_name] = {
                    "status": "failed",
                    "reason": f"Unexpected return type: {type(result)}",
                }

        except Exception as e:
            logger.error(f"✗ {plotter_name}: exception during rendering", exc_info=e)
            status_summary[plotter_name] = {
                "status": "failed",
                "reason": f"Exception: {e}",
            }

    # Include status summary in result
    aggregate["_plot_status"] = status_summary

    return aggregate


from . import plotters as _default_plotters  # noqa: F401,E402
