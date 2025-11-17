"""Tests for ModelService."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import os

# Prevent JAX initialization
os.environ["JAX_PLATFORMS"] = "cpu"

# Import directly to avoid dashboard.__init__ cascade
sys.path.insert(0, '/home/user/active_learning')
sys.path.insert(0, '/home/user/active_learning/src')

from use_cases.dashboard.services.model_service import (
    ModelService,
    CreateModelRequest,
    LoadModelRequest,
)
from use_cases.dashboard.core.model_manager import ModelManager
from rcmvae.domain.config import SSVAEConfig


class TestModelService:
    """Test suite for ModelService."""

    @pytest.fixture
    def temp_models_dir(self):
        """Create a temporary models directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def model_manager(self, temp_models_dir, monkeypatch):
        """Create a model manager with temp directory."""
        monkeypatch.setattr("use_cases.dashboard.core.model_manager.MODELS_DIR", temp_models_dir)
        manager = ModelManager()
        manager.MODELS_DIR = temp_models_dir
        return manager

    @pytest.fixture
    def service(self, model_manager):
        """Create a model service instance."""
        return ModelService(model_manager)

    @patch("use_cases.dashboard.services.model_service.load_mnist_scaled")
    def test_create_model(self, mock_load_mnist, service):
        """Test creating a new model."""
        # Mock MNIST data
        import numpy as np
        mock_load_mnist.return_value = (
            np.random.rand(1000, 28, 28).astype(np.float32),  # x
            np.random.randint(0, 10, 1000).astype(np.int32),  # y
            None,  # x_test
            None,  # y_test
            "test_source"  # source
        )

        request = CreateModelRequest(
            name="test_model",
            config=SSVAEConfig(),
            dataset_total_samples=100,
            dataset_seed=42,
        )

        model_id = service.create_model(request)

        # Verify model_id was returned
        assert model_id is not None
        assert isinstance(model_id, str)

        # Verify model directory was created
        model_dir = service._manager.model_dir(model_id)
        assert model_dir.exists()

        # Verify metadata was saved
        metadata = service._manager.load_metadata(model_id)
        assert metadata is not None
        assert metadata.name == "test_model"
        assert metadata.dataset_total_samples == 100
        assert metadata.dataset_seed == 42

        # Verify config was saved
        config = service._manager.load_config(model_id)
        assert config is not None

        # Verify labels.csv was created
        labels_path = service._manager.labels_path(model_id)
        assert labels_path.exists()

    @patch("use_cases.dashboard.services.model_service.load_mnist_scaled")
    def test_create_model_auto_id(self, mock_load_mnist, service):
        """Test creating model without name generates auto ID."""
        import numpy as np
        mock_load_mnist.return_value = (
            np.random.rand(1000, 28, 28).astype(np.float32),
            np.random.randint(0, 10, 1000).astype(np.int32),
            None,
            None,
            "test_source"
        )

        request = CreateModelRequest(
            name=None,
            config=SSVAEConfig(),
            dataset_total_samples=100,
            dataset_seed=42,
        )

        model_id = service.create_model(request)

        assert model_id.startswith("model_")

    def test_delete_model(self, service, model_manager):
        """Test deleting a model."""
        # Create a model directory
        model_id = "test_model"
        model_dir = model_manager.model_dir(model_id)
        model_dir.mkdir(parents=True)

        # Delete it
        success = service.delete_model(model_id)

        assert success is True
        assert not model_dir.exists()

    def test_delete_nonexistent_model(self, service):
        """Test deleting non-existent model returns False."""
        success = service.delete_model("nonexistent")
        assert success is False

    def test_list_models_empty(self, service):
        """Test listing models when none exist."""
        models = service.list_models()
        assert models == []

    def test_update_metadata(self, service, model_manager):
        """Test updating model metadata."""
        from use_cases.dashboard.core.state_models import ModelMetadata
        from datetime import datetime

        model_id = "test_model"
        model_dir = model_manager.model_dir(model_id)
        model_dir.mkdir(parents=True)

        metadata = ModelMetadata(
            model_id=model_id,
            name="Test Model",
            created_at=datetime.utcnow().isoformat(),
            last_modified=datetime.utcnow().isoformat(),
            dataset="mnist",
            total_epochs=5,
            labeled_count=10,
            latest_loss=0.5,
        )

        service.update_metadata(metadata)

        # Verify it was saved
        loaded = service._manager.load_metadata(model_id)
        assert loaded is not None
        assert loaded.total_epochs == 5
        assert loaded.labeled_count == 10
        assert loaded.latest_loss == 0.5
