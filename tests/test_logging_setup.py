"""Unit tests for logging setup and utilities.

Tests the structured logging system for experiment runs.
"""
import logging
import tempfile
from pathlib import Path

import pytest

from rcmvae.domain.config import SSVAEConfig
from use_cases.experiments.src.logging import setup_experiment_logging
from use_cases.experiments.src.logging.setup import (
    TrainingLogFilter,
    log_section_header,
    log_model_initialization,
    log_training_epoch,
)


class TestLoggingSetup:
    """Test logging initialization and configuration."""

    def test_creates_log_directory(self):
        """Logger should create log directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            assert not log_dir.exists()

            setup_experiment_logging(log_dir)

            assert log_dir.exists()
            assert log_dir.is_dir()

    def test_creates_log_files(self):
        """Logger should create all expected log files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"

            logger = setup_experiment_logging(log_dir)
            logger.info("Test message")

            # Check all log files exist
            assert (log_dir / "experiment.log").exists()
            assert (log_dir / "training.log").exists()
            assert (log_dir / "errors.log").exists()

    def test_logger_has_correct_name(self):
        """Logger should have the specified name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"

            logger = setup_experiment_logging(log_dir, name="test_logger")

            assert logger.name == "test_logger"

    def test_logger_level_is_debug(self):
        """Logger should capture DEBUG level for file handlers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"

            logger = setup_experiment_logging(log_dir)

            assert logger.level == logging.DEBUG

    def test_info_message_in_experiment_log(self):
        """INFO messages should appear in experiment.log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"

            logger = setup_experiment_logging(log_dir)
            logger.info("Test INFO message")

            content = (log_dir / "experiment.log").read_text()
            assert "Test INFO message" in content
            assert "INFO" in content

    def test_debug_message_in_experiment_log(self):
        """DEBUG messages should appear in experiment.log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"

            logger = setup_experiment_logging(log_dir)
            logger.debug("Test DEBUG message")

            content = (log_dir / "experiment.log").read_text()
            assert "Test DEBUG message" in content
            assert "DEBUG" in content

    def test_warning_in_errors_log(self):
        """WARNING messages should appear in errors.log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"

            logger = setup_experiment_logging(log_dir)
            logger.warning("Test WARNING message")

            content = (log_dir / "errors.log").read_text()
            assert "Test WARNING message" in content
            assert "WARNING" in content

    def test_error_in_errors_log(self):
        """ERROR messages should appear in errors.log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"

            logger = setup_experiment_logging(log_dir)
            logger.error("Test ERROR message")

            content = (log_dir / "errors.log").read_text()
            assert "Test ERROR message" in content
            assert "ERROR" in content

    def test_info_not_in_errors_log(self):
        """INFO messages should NOT appear in errors.log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"

            logger = setup_experiment_logging(log_dir)
            logger.info("Test INFO message")

            content = (log_dir / "errors.log").read_text()
            assert "Test INFO message" not in content


class TestTrainingLogFilter:
    """Test the training log filter."""

    def test_filters_epoch_messages(self):
        """Messages containing 'epoch' should pass filter."""
        filter_obj = TrainingLogFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Epoch 5/100 | loss=198.3",
            args=(),
            exc_info=None,
        )

        assert filter_obj.filter(record) is True

    def test_filters_loss_messages(self):
        """Messages containing 'loss' should pass filter."""
        filter_obj = TrainingLogFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Training loss: 198.3",
            args=(),
            exc_info=None,
        )

        assert filter_obj.filter(record) is True

    def test_filters_accuracy_messages(self):
        """Messages containing 'accuracy' should pass filter."""
        filter_obj = TrainingLogFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Accuracy: 0.87",
            args=(),
            exc_info=None,
        )

        assert filter_obj.filter(record) is True

    def test_blocks_non_training_messages(self):
        """Messages without training keywords should be filtered out."""
        filter_obj = TrainingLogFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Model configuration loaded",
            args=(),
            exc_info=None,
        )

        assert filter_obj.filter(record) is False

    def test_case_insensitive(self):
        """Filter should be case-insensitive."""
        filter_obj = TrainingLogFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="EPOCH 5/100",
            args=(),
            exc_info=None,
        )

        assert filter_obj.filter(record) is True


class TestLoggingHelpers:
    """Test logging helper functions."""

    def test_log_section_header(self):
        """Section headers should be formatted correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            logger = setup_experiment_logging(log_dir)

            log_section_header(logger, "Test Section")

            content = (log_dir / "experiment.log").read_text()
            assert "Test Section" in content
            assert "═" * 80 in content

    def test_log_model_initialization(self):
        """Model initialization should log architecture details."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            logger = setup_experiment_logging(log_dir)

            config = SSVAEConfig(
                prior_type="mixture",
                num_components=10,
                use_tau_classifier=True,
                use_component_aware_decoder=True,
                use_heteroscedastic_decoder=True,
            )

            log_model_initialization(logger, config, "mix10_tau_ca-het")

            content = (log_dir / "experiment.log").read_text()
            assert "Model Initialization" in content
            assert "mix10_tau_ca-het" in content
            assert "mixture" in content
            assert "τ-classifier" in content
            assert "Component-aware: ✓" in content
            assert "Heteroscedastic: ✓" in content

    def test_log_training_epoch(self):
        """Training epoch should log losses and metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            logger = setup_experiment_logging(log_dir)

            losses = {
                "loss.total": 198.3,
                "loss.recon": 185.2,
                "loss.kl_z": 12.1,
            }
            metrics = {"acc": 0.87}

            log_training_epoch(logger, epoch=5, max_epochs=100, losses=losses, metrics=metrics)

            content = (log_dir / "experiment.log").read_text()
            assert "Epoch   5/100" in content
            assert "198.3" in content or "198.30" in content
            assert "0.87" in content or "0.870" in content

    def test_training_epoch_in_training_log(self):
        """Training epochs should appear in training.log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            logger = setup_experiment_logging(log_dir)

            losses = {"loss.total": 198.3}
            log_training_epoch(logger, epoch=5, max_epochs=100, losses=losses)

            content = (log_dir / "training.log").read_text()
            assert "Epoch" in content
            assert "loss" in content


class TestLoggingIntegration:
    """Test logging in realistic scenarios."""

    def test_full_experiment_logging(self):
        """Test complete experiment logging workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            logger = setup_experiment_logging(log_dir)

            # Experiment start
            log_section_header(logger, "Experiment Start")
            logger.info("Run ID: a3f9c2d1")

            # Model initialization
            config = SSVAEConfig(prior_type="mixture", num_components=10)
            log_model_initialization(logger, config, "mix10_head_plain")

            # Training
            log_section_header(logger, "Training")
            for epoch in range(1, 4):
                losses = {"loss.total": 200.0 - epoch}
                log_training_epoch(logger, epoch, 100, losses)

            # Warnings
            logger.warning("Test warning")

            # Completion
            log_section_header(logger, "Complete")
            logger.info("Training finished")

            # Verify all log files have content
            exp_content = (log_dir / "experiment.log").read_text()
            train_content = (log_dir / "training.log").read_text()
            error_content = (log_dir / "errors.log").read_text()

            assert "Run ID" in exp_content
            assert "Model Initialization" in exp_content
            assert "Epoch" in train_content
            assert "Test warning" in error_content
