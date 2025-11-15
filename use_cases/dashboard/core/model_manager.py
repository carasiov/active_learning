"""Model persistence layer - handles save/load to disk."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional, Dict, Any
import json

from use_cases.dashboard.core.state_models import ModelMetadata, TrainingHistory
from rcmvae.domain.config import SSVAEConfig

MODELS_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "models"


class ModelManager:
    """Handles model persistence to disk."""
    
    @staticmethod
    def ensure_models_dir() -> None:
        """Create models directory if missing."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def generate_model_id() -> str:
        """Generate unique model ID like 'model_001'."""
        ModelManager.ensure_models_dir()
        existing = [d.name for d in MODELS_DIR.iterdir() if d.is_dir()]
        
        # Extract numbers from existing IDs
        numbers = []
        for name in existing:
            if name.startswith("model_"):
                try:
                    num = int(name.split("_")[1])
                    numbers.append(num)
                except (IndexError, ValueError):
                    pass
        
        next_num = max(numbers, default=0) + 1
        return f"model_{next_num:03d}"
    
    @staticmethod
    def model_dir(model_id: str) -> Path:
        """Get directory path for model."""
        return MODELS_DIR / model_id
    
    @staticmethod
    def create_model_directory(model_id: str) -> Path:
        """Create model directory structure."""
        model_path = ModelManager.model_dir(model_id)
        model_path.mkdir(parents=True, exist_ok=True)
        return model_path
    
    @staticmethod
    def save_metadata(metadata: ModelMetadata) -> None:
        """Save metadata.json."""
        path = ModelManager.model_dir(metadata.model_id) / "metadata.json"
        with open(path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)
    
    @staticmethod
    def load_metadata(model_id: str) -> Optional[ModelMetadata]:
        """Load metadata.json."""
        path = ModelManager.model_dir(model_id) / "metadata.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return ModelMetadata.from_dict(data)
    
    @staticmethod
    def save_history(model_id: str, history: TrainingHistory) -> None:
        """Save training history to history.json."""
        path = ModelManager.model_dir(model_id) / "history.json"
        data = {
            "epochs": list(history.epochs),
            "train_loss": list(history.train_loss),
            "val_loss": list(history.val_loss),
            "train_reconstruction_loss": list(history.train_reconstruction_loss),
            "val_reconstruction_loss": list(history.val_reconstruction_loss),
            "train_kl_loss": list(history.train_kl_loss),
            "val_kl_loss": list(history.val_kl_loss),
            "train_classification_loss": list(history.train_classification_loss),
            "val_classification_loss": list(history.val_classification_loss),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def load_history(model_id: str) -> TrainingHistory:
        """Load training history from history.json."""
        path = ModelManager.model_dir(model_id) / "history.json"
        if not path.exists():
            return TrainingHistory.empty()
        
        with open(path, "r") as f:
            data = json.load(f)
        
        return TrainingHistory(
            epochs=data.get("epochs", []),
            train_loss=data.get("train_loss", []),
            val_loss=data.get("val_loss", []),
            train_reconstruction_loss=data.get("train_reconstruction_loss", []),
            val_reconstruction_loss=data.get("val_reconstruction_loss", []),
            train_kl_loss=data.get("train_kl_loss", []),
            val_kl_loss=data.get("val_kl_loss", []),
            train_classification_loss=data.get("train_classification_loss", []),
            val_classification_loss=data.get("val_classification_loss", []),
        )
    
    @staticmethod
    def checkpoint_path(model_id: str) -> Path:
        """Get checkpoint path for model."""
        return ModelManager.model_dir(model_id) / "checkpoint.ckpt"
    
    @staticmethod
    def labels_path(model_id: str) -> Path:
        """Get labels CSV path for model."""
        return ModelManager.model_dir(model_id) / "labels.csv"
    
    @staticmethod
    def config_path(model_id: str) -> Path:
        """Get configuration file path for model."""
        return ModelManager.model_dir(model_id) / "config.json"
    
    @staticmethod
    def save_config(model_id: str, config: SSVAEConfig) -> None:
        """Persist model configuration to config.json."""
        path = ModelManager.config_path(model_id)
        data = asdict(config)
        # Convert tuples to lists for JSON compatibility
        if isinstance(data.get("hidden_dims"), tuple):
            data["hidden_dims"] = list(data["hidden_dims"])
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def load_config(model_id: str) -> Optional[SSVAEConfig]:
        """Load model configuration from disk."""
        path = ModelManager.config_path(model_id)
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        # Normalize list fields back to tuples
        hidden_dims = data.get("hidden_dims")
        if hidden_dims is not None and not isinstance(hidden_dims, tuple):
            data["hidden_dims"] = tuple(int(dim) for dim in hidden_dims)
        return SSVAEConfig(**data)
    
    @staticmethod
    def list_all_models() -> Dict[str, ModelMetadata]:
        """Scan models directory and load all metadata."""
        ModelManager.ensure_models_dir()
        models = {}
        
        for model_dir in MODELS_DIR.iterdir():
            if not model_dir.is_dir():
                continue
            
            metadata = ModelManager.load_metadata(model_dir.name)
            if metadata:
                models[metadata.model_id] = metadata
        
        return models
    
    @staticmethod
    def delete_model(model_id: str) -> None:
        """Delete model directory and all contents."""
        import shutil
        model_path = ModelManager.model_dir(model_id)
        if model_path.exists():
            shutil.rmtree(model_path)
