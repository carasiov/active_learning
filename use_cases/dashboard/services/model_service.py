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

    def create_model(self, request: CreateModelRequest) -> str:
        """Create and initialize new model.

        Args:
            request: Model creation specification

        Returns:
            model_id: ID of created model

        Raises:
            ValidationError: If config invalid
        """
        with self._lock:
            from datetime import datetime
            from use_cases.dashboard.core.state_models import TrainingHistory
            import pandas as pd
            import numpy as np
            from use_cases.experiments.data.mnist.mnist import load_mnist_scaled
            import re

            # Generate model ID from name or auto-generate
            if request.name and request.name.strip():
                sanitized = re.sub(r'[^a-z0-9_-]', '_', request.name.strip().lower())
                sanitized = re.sub(r'_+', '_', sanitized)
                model_id = sanitized.strip('_') or "model"
                base_id = model_id
                counter = 1
                while self._manager.model_dir(model_id).exists():
                    model_id = f"{base_id}_{counter}"
                    counter += 1
            else:
                model_id = self._manager.generate_model_id()

            display_name = request.name.strip() if request.name and request.name.strip() else model_id

            # Create directory
            self._manager.create_model_directory(model_id)

            # Load and snapshot dataset
            total_samples = request.dataset_total_samples
            rng_seed = request.dataset_seed
            rng = np.random.default_rng(rng_seed)

            x_full, y_full, _, _, source = load_mnist_scaled(
                reshape=True,
                hw=(28, 28),
                dtype=np.float32,
            )

            max_available = x_full.shape[0]
            if total_samples > max_available:
                total_samples = max_available

            selected_indices = rng.choice(max_available, size=total_samples, replace=False)
            selected_indices = selected_indices.tolist()

            # Save dataset configuration
            dataset_config = {
                "dataset": "mnist",
                "source": source,
                "indices": selected_indices,
                "labeled_positions": [],  # Will be populated by labeling service
                "seed": rng_seed,
                "total_samples": total_samples,
                "labeled_samples": 0,
            }
            self._manager.save_dataset_config(model_id, dataset_config)

            # Initialize empty labels
            labels_path = self._manager.labels_path(model_id)
            df = pd.DataFrame(columns=["Serial", "label"])
            df.to_csv(labels_path, index=False)

            # Create metadata
            now = datetime.utcnow().isoformat()
            metadata = ModelMetadata(
                model_id=model_id,
                name=display_name,
                created_at=now,
                last_modified=now,
                dataset="mnist",
                total_epochs=0,
                labeled_count=0,
                latest_loss=None,
                dataset_total_samples=total_samples,
                dataset_seed=rng_seed,
            )

            # Save metadata and auxiliary files
            self._manager.save_metadata(metadata)
            self._manager.save_history(model_id, TrainingHistory.empty())
            self._manager.save_config(model_id, request.config)

            return model_id

    def load_model(self, request: LoadModelRequest) -> Optional[ModelState]:
        """Load model from disk.

        Args:
            request: Load specification

        Returns:
            ModelState or None if not found
        """
        with self._lock:
            from use_cases.dashboard.core.state_models import (
                DataState,
                TrainingStatus,
                TrainingState,
                UIState,
            )
            from use_cases.dashboard.utils.visualization import _build_hover_metadata
            from use_cases.experiments.data.mnist.mnist import load_mnist_scaled
            from rcmvae.application.model_api import SSVAE
            from rcmvae.domain.config import SSVAEConfig
            from rcmvae.application.runtime.interactive import InteractiveTrainer
            from use_cases.dashboard.core.model_runs import load_run_records
            import pandas as pd
            import numpy as np
            from dataclasses import replace as dc_replace

            # Load metadata
            metadata = self._manager.load_metadata(request.model_id)
            if metadata is None:
                return None

            # Load config
            config = self._manager.load_config(request.model_id)
            if config is None:
                config = SSVAEConfig()

            # Initialize model
            model = SSVAE(input_dim=(28, 28), config=config)
            checkpoint_path = self._manager.checkpoint_path(request.model_id)
            if checkpoint_path.exists():
                model.load_model_weights(str(checkpoint_path))
                model.weights_path = str(checkpoint_path)

            trainer = InteractiveTrainer(model)

            # Load dataset
            dataset_config = self._manager.load_dataset_config(request.model_id)
            if dataset_config:
                indices = np.asarray(dataset_config.get("indices", []), dtype=np.int64)
                if indices.size == 0:
                    raise ValueError("Dataset configuration has no indices")

                x_full, y_full, _, _, _ = load_mnist_scaled(
                    reshape=True,
                    hw=(28, 28),
                    dtype=np.float32,
                )

                max_available = x_full.shape[0]
                if np.any(indices >= max_available):
                    raise ValueError("Dataset indices exceed available MNIST samples")

                x_train = x_full[indices]
                true_labels = y_full[indices].astype(np.int32)
            else:
                # Fallback for backward compatibility
                return None

            # Load labels
            labels_array = np.full(shape=(x_train.shape[0],), fill_value=np.nan, dtype=float)
            labels_path = self._manager.labels_path(request.model_id)
            if labels_path.exists():
                stored_labels = pd.read_csv(labels_path)
                if not stored_labels.empty and "Serial" in stored_labels.columns:
                    stored_labels["Serial"] = pd.to_numeric(stored_labels["Serial"], errors="coerce")
                    stored_labels = stored_labels.dropna(subset=["Serial"])
                    stored_labels["Serial"] = stored_labels["Serial"].astype(int)
                    stored_labels["label"] = pd.to_numeric(stored_labels.get("label"), errors="coerce").astype("Int64")
                    serials = stored_labels["Serial"].to_numpy()
                    label_values = stored_labels["label"].astype(int).to_numpy()
                    valid_mask = (serials >= 0) & (serials < x_train.shape[0])
                    labels_array[serials[valid_mask]] = label_values[valid_mask].astype(float)

            labeled_count = int(np.sum(~np.isnan(labels_array)))
            metadata = dc_replace(
                metadata,
                labeled_count=labeled_count,
                dataset_total_samples=x_train.shape[0],
            )

            # Get predictions
            latent, recon, pred_classes, pred_certainty = model.predict(x_train)
            hover_metadata = _build_hover_metadata(pred_classes, pred_certainty, labels_array, true_labels)

            # Load history and runs
            history = self._manager.load_history(request.model_id)
            run_records = load_run_records(request.model_id)

            # Build state
            data_state = DataState(
                x_train=x_train,
                labels=labels_array,
                true_labels=true_labels,
                latent=latent,
                reconstructed=recon,
                pred_classes=pred_classes,
                pred_certainty=pred_certainty,
                hover_metadata=hover_metadata,
                version=0
            )

            training_status = TrainingStatus(
                state=TrainingState.IDLE,
                target_epochs=0,
                status_messages=[],
                thread=None
            )

            ui_state = UIState(
                selected_sample=0,
                color_mode="user_labels"
            )

            model_state = ModelState(
                model_id=request.model_id,
                metadata=metadata,
                model=model,
                trainer=trainer,
                config=model.config,
                data=data_state,
                training=training_status,
                ui=ui_state,
                history=history,
                runs=tuple(run_records)
            )

            return model_state

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
