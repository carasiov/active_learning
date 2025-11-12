"""Experiment orchestration pipeline (config, data prep, training).

Includes metadata augmentation for self-documenting experiments.
"""

from .config import add_repo_paths, augment_config_metadata, load_experiment_config
from .data import prepare_data
from .train import run_training_pipeline

__all__ = [
    "add_repo_paths",
    "augment_config_metadata",
    "load_experiment_config",
    "prepare_data",
    "run_training_pipeline",
]
