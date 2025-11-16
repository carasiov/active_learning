"""Tests for TrainingService."""

import pytest
from queue import Queue
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import sys
import os

# Prevent JAX initialization
os.environ["JAX_PLATFORMS"] = "cpu"

# Import directly to avoid dashboard.__init__ cascade
sys.path.insert(0, '/home/user/active_learning')
sys.path.insert(0, '/home/user/active_learning/src')

from use_cases.dashboard.services.training_service import (
    InProcessTrainingService,
    TrainingJob,
    JobStatus,
    AlreadyTrainingError,
)

# Mock InteractiveTrainer to avoid JAX
InteractiveTrainer = MagicMock


class TestInProcessTrainingService:
    """Test suite for InProcessTrainingService."""

    @pytest.fixture
    def metrics_queue(self):
        """Create a metrics queue for testing."""
        return Queue()

    @pytest.fixture
    def service(self, metrics_queue):
        """Create a training service instance."""
        return InProcessTrainingService(metrics_queue)

    @pytest.fixture
    def mock_trainer(self):
        """Create a mock trainer."""
        trainer = Mock(spec=InteractiveTrainer)
        trainer.train = Mock()
        return trainer

    def test_service_initialization(self, service):
        """Test service initializes with empty state."""
        assert len(service._jobs) == 0
        assert len(service._threads) == 0
        assert len(service._stop_flags) == 0

    def test_start_training_creates_job(self, service, mock_trainer):
        """Test starting training creates a job entry."""
        job = TrainingJob(
            job_id="test_job_1",
            model_id="test_model",
            trainer=mock_trainer,
            num_epochs=5,
            x_train=np.zeros((10, 28, 28)),
            labels=np.zeros(10),
            checkpoint_path="/tmp/test.ckpt",
            run_epoch_offset=0,
        )

        job_id = service.start_training(job)

        assert job_id == "test_job_1"
        assert job_id in service._jobs
        assert service._jobs[job_id].status == JobStatus.QUEUED

    def test_already_training_error(self, service, mock_trainer):
        """Test error when starting training on already-training model."""
        job1 = TrainingJob(
            job_id="job1",
            model_id="model1",
            trainer=mock_trainer,
            num_epochs=5,
            x_train=np.zeros((10, 28, 28)),
            labels=np.zeros(10),
            checkpoint_path="/tmp/test.ckpt",
        )

        service.start_training(job1)

        # Try to start another job for same model
        job2 = TrainingJob(
            job_id="job2",
            model_id="model1",  # Same model
            trainer=mock_trainer,
            num_epochs=5,
            x_train=np.zeros((10, 28, 28)),
            labels=np.zeros(10),
            checkpoint_path="/tmp/test.ckpt",
        )

        with pytest.raises(AlreadyTrainingError):
            service.start_training(job2)

    def test_get_progress(self, service, mock_trainer):
        """Test getting job progress."""
        job = TrainingJob(
            job_id="test_job",
            model_id="test_model",
            trainer=mock_trainer,
            num_epochs=5,
            x_train=np.zeros((10, 28, 28)),
            labels=np.zeros(10),
            checkpoint_path="/tmp/test.ckpt",
        )

        service.start_training(job)
        progress = service.get_progress("test_job")

        assert progress is not None
        assert progress.status in {JobStatus.QUEUED, JobStatus.RUNNING}
        assert progress.model_id == "test_model"

    def test_get_progress_nonexistent(self, service):
        """Test getting progress for non-existent job returns None."""
        progress = service.get_progress("nonexistent")
        assert progress is None

    def test_stop_training(self, service, mock_trainer):
        """Test stopping a training job."""
        job = TrainingJob(
            job_id="test_job",
            model_id="test_model",
            trainer=mock_trainer,
            num_epochs=5,
            x_train=np.zeros((10, 28, 28)),
            labels=np.zeros(10),
            checkpoint_path="/tmp/test.ckpt",
        )

        service.start_training(job)
        success = service.stop_training("test_job")

        assert success is True
        assert service._stop_flags["test_job"].is_set()

    def test_stop_nonexistent_job(self, service):
        """Test stopping non-existent job returns False."""
        success = service.stop_training("nonexistent")
        assert success is False

    def test_is_training(self, service, mock_trainer):
        """Test checking if model is training."""
        assert service.is_training("model1") is False

        job = TrainingJob(
            job_id="job1",
            model_id="model1",
            trainer=mock_trainer,
            num_epochs=5,
            x_train=np.zeros((10, 28, 28)),
            labels=np.zeros(10),
            checkpoint_path="/tmp/test.ckpt",
        )

        service.start_training(job)
        assert service.is_training("model1") is True
