"""ExperimentService orchestrates end-to-end training runs."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from rcmvae.application.model_api import DEFAULT_CHECKPOINT_PATH, SSVAE
from rcmvae.domain.config import SSVAEConfig


ArrayLike = np.ndarray | Tuple[np.ndarray, np.ndarray]


@dataclass(slots=True)
class TrainingArtifacts:
    """Outputs from a single training run."""

    model: SSVAE
    history: Dict[str, Any]
    latent: np.ndarray
    reconstructions: ArrayLike
    predictions: np.ndarray
    certainty: np.ndarray
    responsibilities: Optional[np.ndarray]
    pi_values: Optional[np.ndarray]
    train_time: float
    diagnostics_dir: Optional[Path]


class ExperimentService:
    """High-level orchestration for experiments."""

    def __init__(self, input_dim: Tuple[int, int] = (28, 28)):
        self.input_dim = input_dim

    def run(
        self,
        config: SSVAEConfig,
        x_train: np.ndarray,
        y_train: np.ndarray,
        *,
        weights_path: Path | str | None,
        export_history: bool = True,
    ) -> TrainingArtifacts:
        """Train an SSVAE instance and return evaluation artifacts."""
        x_train = np.asarray(x_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32).reshape((-1,))

        model = SSVAE(input_dim=self.input_dim, config=config)

        ckpt_path = str(weights_path) if weights_path is not None else str(DEFAULT_CHECKPOINT_PATH)
        start = time.time()
        history = model.fit(x_train, y_train, weights_path=ckpt_path, export_history=export_history)
        train_time = time.time() - start

        latent, recon, predictions, certainty = model.predict_batched(x_train)

        responsibilities = None
        pi_values = None
        if config.is_mixture_based_prior():
            try:
                (
                    _,
                    _,
                    _,
                    _,
                    responsibilities,
                    pi_values,
                ) = model.predict_batched(x_train, return_mixture=True)
            except TypeError:
                responsibilities = None
                pi_values = None

        diagnostics_dir = model.last_diagnostics_dir

        return TrainingArtifacts(
            model=model,
            history=history,
            latent=latent,
            reconstructions=recon,
            predictions=predictions,
            certainty=certainty,
            responsibilities=responsibilities,
            pi_values=pi_values,
            train_time=train_time,
            diagnostics_dir=diagnostics_dir,
        )
