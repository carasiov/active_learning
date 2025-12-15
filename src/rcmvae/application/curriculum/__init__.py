"""Curriculum learning module for channel unlocking ("pots")."""

from rcmvae.application.curriculum.controller import (
    CurriculumController,
    CurriculumConfig,
    CurriculumState,
    UnlockEvent,
)
from rcmvae.application.curriculum.hooks import build_curriculum_hooks

__all__ = [
    "CurriculumController",
    "CurriculumConfig",
    "CurriculumState",
    "UnlockEvent",
    "build_curriculum_hooks",
]
