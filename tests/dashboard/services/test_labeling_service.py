"""Tests for LabelingService."""

import pytest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import sys

# Prevent JAX initialization
import os
os.environ["JAX_PLATFORMS"] = "cpu"

# Import directly to avoid dashboard.__init__ cascade
sys.path.insert(0, '/home/user/active_learning')
from use_cases.dashboard.services.labeling_service import LabelingService
from use_cases.dashboard.core.model_manager import ModelManager


class TestLabelingService:
    """Test suite for LabelingService."""

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
        """Create a labeling service instance."""
        return LabelingService(model_manager)

    @pytest.fixture
    def test_model(self, model_manager):
        """Create a test model directory."""
        model_id = "test_model"
        model_dir = model_manager.model_dir(model_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_id

    def test_set_label_creates_csv(self, service, test_model):
        """Test setting a label creates labels.csv."""
        success = service.set_label(test_model, sample_idx=0, label=5)

        assert success is True

        # Check CSV was created
        labels_path = service._manager.model_dir(test_model) / "labels.csv"
        assert labels_path.exists()

        df = pd.read_csv(labels_path, index_col=0)
        assert 0 in df.index
        assert df.loc[0, "label"] == 5

    def test_set_multiple_labels(self, service, test_model):
        """Test setting multiple labels."""
        service.set_label(test_model, 0, 1)
        service.set_label(test_model, 5, 3)
        service.set_label(test_model, 10, 7)

        labels_path = service._manager.model_dir(test_model) / "labels.csv"
        df = pd.read_csv(labels_path, index_col=0)

        assert len(df) == 3
        assert df.loc[0, "label"] == 1
        assert df.loc[5, "label"] == 3
        assert df.loc[10, "label"] == 7

    def test_delete_label(self, service, test_model):
        """Test deleting a label (setting to None)."""
        service.set_label(test_model, 0, 5)
        service.set_label(test_model, 1, 7)

        # Delete first label
        service.set_label(test_model, 0, None)

        labels_path = service._manager.model_dir(test_model) / "labels.csv"
        df = pd.read_csv(labels_path, index_col=0)

        assert 0 not in df.index
        assert 1 in df.index
        assert df.loc[1, "label"] == 7

    def test_invalid_label_raises_error(self, service, test_model):
        """Test invalid label raises ValueError."""
        with pytest.raises(ValueError, match="Invalid label"):
            service.set_label(test_model, 0, 10)  # Must be 0-9

        with pytest.raises(ValueError, match="Invalid label"):
            service.set_label(test_model, 0, -1)

    def test_get_label(self, service, test_model):
        """Test getting a label."""
        service.set_label(test_model, 5, 7)

        label = service.get_label(test_model, 5)
        assert label == 7

    def test_get_label_unlabeled(self, service, test_model):
        """Test getting label for unlabeled sample returns None."""
        label = service.get_label(test_model, 999)
        assert label is None

    def test_get_labeled_count(self, service, test_model):
        """Test getting labeled count."""
        assert service.get_labeled_count(test_model) == 0

        service.set_label(test_model, 0, 1)
        service.set_label(test_model, 1, 2)
        service.set_label(test_model, 2, 3)

        assert service.get_labeled_count(test_model) == 3

    def test_get_labeled_count_no_csv(self, service, test_model):
        """Test labeled count is 0 when no CSV exists."""
        count = service.get_labeled_count(test_model)
        assert count == 0
