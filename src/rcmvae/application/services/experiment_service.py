"""ExperimentService orchestrates end-to-end training runs."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rcmvae.application.model_api import DEFAULT_CHECKPOINT_PATH, SSVAE
from rcmvae.domain.config import SSVAEConfig


ArrayLike = np.ndarray | Tuple[np.ndarray, np.ndarray]


@dataclass
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
    # Curriculum outputs (None if curriculum disabled)
    curriculum_summary: Optional[Dict[str, Any]] = None
    curriculum_history: Optional[List[Dict[str, Any]]] = None


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
        curriculum_config: Dict[str, Any] | None = None,
    ) -> TrainingArtifacts:
        """Train an SSVAE instance and return evaluation artifacts.

        Args:
            config: SSVAE model configuration
            x_train: Training images [N, H, W]
            y_train: Training labels [N] (NaN for unlabeled)
            weights_path: Path to save best checkpoint
            export_history: Whether to export history CSV and plots
            curriculum_config: Optional curriculum configuration dict from YAML.
                              If provided and enabled=True, uses curriculum learning.

        Returns:
            TrainingArtifacts with model, history, predictions, and curriculum info
        """
        from rcmvae.application.curriculum import (
            CurriculumConfig,
            CurriculumController,
            build_curriculum_hooks,
        )

        x_train = np.asarray(x_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32).reshape((-1,))

        model = SSVAE(input_dim=self.input_dim, config=config)

        # Set up curriculum if configured
        curriculum_ctrl = None
        curriculum_hooks = None
        if curriculum_config:
            curr_cfg = CurriculumConfig.from_dict(curriculum_config)
            if curr_cfg.enabled:
                print(f"\n[Curriculum] Enabled with k_active_init={curr_cfg.k_active_init}")
                curriculum_ctrl = CurriculumController(curr_cfg, k_max=config.num_components)
                curriculum_hooks = build_curriculum_hooks(curriculum_ctrl)

        ckpt_path = str(weights_path) if weights_path is not None else str(DEFAULT_CHECKPOINT_PATH)
        start = time.time()
        history = model.fit(
            x_train, y_train,
            weights_path=ckpt_path,
            export_history=export_history,
            external_hooks=curriculum_hooks,
        )
        train_time = time.time() - start

        # Get curriculum summary if enabled
        curriculum_summary = None
        curriculum_history = None
        if curriculum_ctrl is not None:
            curriculum_summary = curriculum_ctrl.get_summary()
            curriculum_history = curriculum_ctrl.get_epoch_history()
            print(f"[Curriculum] Final: k_active={curriculum_summary['final_k_active']}, "
                  f"unlocks={curriculum_summary['unlock_count']}")

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
            curriculum_summary=curriculum_summary,
            curriculum_history=curriculum_history,
        )
