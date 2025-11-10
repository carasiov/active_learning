"""Visualization registry for experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

PlotResult = Dict[str, Any]
Plotter = Callable[["VisualizationContext"], Optional[PlotResult]]


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
    context.figures_dir.mkdir(parents=True, exist_ok=True)
    aggregate: PlotResult = {}
    for plotter in _PLOTTERS:
        result = plotter(context)
        if not result:
            continue
        aggregate.update(result)
    return aggregate


from . import plotters as _default_plotters  # noqa: F401,E402
