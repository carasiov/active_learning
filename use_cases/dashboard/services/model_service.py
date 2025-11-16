"""Model lifecycle management service."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path
import threading

from rcmvae.application.runtime.interactive import InteractiveTrainer
from rcmvae.application.model_api import SSVAE
from rcmvae.domain.config import SSVAEConfig

from use_cases.dashboard.core.state_models import ModelState, ModelMetadata
from use_cases.dashboard.core.model_manager import ModelManager


@dataclass
class CreateModelRequest:
    """Request to create new model."""
    name: str
    config: SSVAEConfig
    dataset_total_samples: int
    dataset_seed: int


@dataclass
class LoadModelRequest:
    """Request to load existing model."""
    model_id: str


class ModelService:
    """Service for model CRUD operations.

    Manages:
    - Model creation and initialization
    - Loading models from disk
    - Model metadata updates
    - Model deletion
    """

    def __init__(self, model_manager: ModelManager):
        """Initialize service.

        Args:
            model_manager: Persistence layer for models
        """
        self._manager = model_manager
        self._lock = threading.Lock()

    def create_model(self, request: CreateModelRequest) -> ModelState:
        """Create and initialize new model.

        Args:
            request: Model creation specification

        Returns:
            Initialized ModelState

        Raises:
            ValidationError: If config invalid
        """
        # TODO: Implement in Phase 1 Session 3
        raise NotImplementedError("ModelService.create_model not yet implemented")

    def load_model(self, request: LoadModelRequest) -> Optional[ModelState]:
        """Load model from disk.

        Args:
            request: Load specification

        Returns:
            ModelState or None if not found
        """
        # TODO: Implement in Phase 1 Session 3
        raise NotImplementedError("ModelService.load_model not yet implemented")

    def delete_model(self, model_id: str) -> bool:
        """Delete model from disk.

        Args:
            model_id: Model identifier

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            model_dir = self._manager.model_dir(model_id)
            if not model_dir.exists():
                return False

            import shutil
            shutil.rmtree(model_dir)
            return True

    def list_models(self) -> List[ModelMetadata]:
        """List all models.

        Returns:
            List of model metadata
        """
        with self._lock:
            models = []
            for model_dir in self._manager.MODELS_DIR.iterdir():
                if model_dir.is_dir():
                    metadata = self._manager.load_metadata(model_dir.name)
                    if metadata:
                        models.append(metadata)

            # Sort by last_modified descending
            models.sort(key=lambda m: m.last_modified, reverse=True)
            return models

    def update_metadata(self, metadata: ModelMetadata) -> None:
        """Update model metadata.

        Args:
            metadata: Updated metadata
        """
        with self._lock:
            self._manager.save_metadata(metadata)
