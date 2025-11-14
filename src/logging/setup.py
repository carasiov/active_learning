"""Logging configuration for experiment runs.

Provides structured logging with multiple output handlers:
1. Console (stdout) - Clean, timestamp-free for monitoring
2. experiment.log - Full detailed log (DEBUG level)
3. training.log - Training progress only (filtered)
4. errors.log - Errors and warnings only

Design principles (from AGENTS.md):
- Important messages to both stdout and file
- Clean terminal output (no clutter)
- Detailed file logs for debugging
- Persistent logs survive terminal disconnection

Usage:
    from logging import setup_experiment_logging

    logger = setup_experiment_logging(run_paths.logs)
    logger.info("Experiment starting...")
    logger.debug("Detailed debug info...")
    logger.error("Something went wrong")
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


class TrainingLogFilter(logging.Filter):
    """Filter to extract only training-related messages.

    Training messages are identified by keywords in the log message:
    - 'epoch'
    - 'loss'
    - 'accuracy'
    - 'learning rate' / 'lr'
    - 'batch'

    This allows separating training progress from other log messages.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Return True if record should be included in training log."""
        msg_lower = record.getMessage().lower()
        keywords = ["epoch", "loss", "accuracy", "learning rate", "lr", "batch"]
        return any(keyword in msg_lower for keyword in keywords)


def setup_experiment_logging(
    log_dir: Path,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    name: str = "experiment",
) -> logging.Logger:
    """Configure logging for experiment runs.

    Creates a logger with multiple handlers for different purposes:
    - Console: Clean output for monitoring (INFO level by default)
    - experiment.log: Complete detailed log (DEBUG level)
    - training.log: Training progress only (filtered INFO level)
    - errors.log: Errors and warnings only (WARNING+ level)

    Args:
        log_dir: Directory to write log files
        console_level: Logging level for console output (default: INFO)
        file_level: Logging level for file output (default: DEBUG)
        name: Logger name (default: "experiment")

    Returns:
        Configured logger instance

    Example:
        >>> from pathlib import Path
        >>> log_dir = Path("results/baseline_20251112_143022/logs")
        >>> logger = setup_experiment_logging(log_dir)
        >>> logger.info("Starting training...")
        Starting training...
        >>> # Also written to experiment.log with timestamp
    """
    # Create log directory
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture everything, handlers filter

    # Clear existing handlers (avoid duplicates on re-initialization)
    logger.handlers.clear()

    # Prevent propagation to root logger (avoid duplicate messages)
    logger.propagate = False

    # ═════════════════════════════════════════════════════════════════════════
    # Formatters
    # ═════════════════════════════════════════════════════════════════════════

    # File formatter: Detailed with timestamps
    file_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console formatter: Clean, no timestamps
    console_formatter = logging.Formatter(fmt="%(message)s")

    # ═════════════════════════════════════════════════════════════════════════
    # Handler 1: Console (stdout) - Clean output
    # ═════════════════════════════════════════════════════════════════════════

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # ═════════════════════════════════════════════════════════════════════════
    # Handler 2: experiment.log - Complete detailed log
    # ═════════════════════════════════════════════════════════════════════════

    experiment_log_path = log_dir / "experiment.log"
    experiment_handler = logging.FileHandler(experiment_log_path, mode="w")
    experiment_handler.setLevel(file_level)
    experiment_handler.setFormatter(file_formatter)
    logger.addHandler(experiment_handler)

    # ═════════════════════════════════════════════════════════════════════════
    # Handler 3: training.log - Training progress only
    # ═════════════════════════════════════════════════════════════════════════

    training_log_path = log_dir / "training.log"
    training_handler = logging.FileHandler(training_log_path, mode="w")
    training_handler.setLevel(logging.INFO)
    training_handler.setFormatter(file_formatter)
    training_handler.addFilter(TrainingLogFilter())
    logger.addHandler(training_handler)

    # ═════════════════════════════════════════════════════════════════════════
    # Handler 4: errors.log - Errors and warnings only
    # ═════════════════════════════════════════════════════════════════════════

    error_log_path = log_dir / "errors.log"
    error_handler = logging.FileHandler(error_log_path, mode="w")
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(file_formatter)
    logger.addHandler(error_handler)

    # Log initialization message
    logger.debug(f"Logging initialized: {log_dir}")
    logger.debug(f"Console level: {logging.getLevelName(console_level)}")
    logger.debug(f"File level: {logging.getLevelName(file_level)}")

    return logger


def log_section_header(logger: logging.Logger, title: str, width: int = 80) -> None:
    """Log a section header with visual separation.

    Creates a centered section header with separator lines for visual clarity.

    Args:
        logger: Logger instance
        title: Section title text
        width: Total width of the header (default: 80 columns)

    Example:
        >>> log_section_header(logger, "Model Configuration")
        ════════════════════════════════════════════════════════════════════════════════
                                  Model Configuration
        ════════════════════════════════════════════════════════════════════════════════
    """
    separator = "═" * width
    logger.info("")
    logger.info(separator)
    logger.info(title.center(width))
    logger.info(separator)


def log_config_summary(logger: logging.Logger, config: dict) -> None:
    """Log configuration summary in a readable format.

    Args:
        logger: Logger instance
        config: Configuration dictionary (can be nested)

    Example:
        >>> config = {"model": {"latent_dim": 2, "num_components": 10}}
        >>> log_config_summary(logger, config)
        Configuration:
          model:
            latent_dim: 2
            num_components: 10
    """
    logger.info("Configuration:")
    _log_dict_recursive(logger, config, indent=1)


def _log_dict_recursive(
    logger: logging.Logger,
    data: dict,
    indent: int = 0,
    max_depth: int = 4,
) -> None:
    """Recursively log nested dictionaries with indentation.

    Args:
        logger: Logger instance
        data: Dictionary to log
        indent: Current indentation level
        max_depth: Maximum nesting depth to display
    """
    if indent > max_depth:
        logger.info("  " * indent + "...")
        return

    for key, value in data.items():
        if isinstance(value, dict):
            logger.info("  " * indent + f"{key}:")
            _log_dict_recursive(logger, value, indent + 1, max_depth)
        elif isinstance(value, (list, tuple)) and len(value) > 10:
            # Summarize long lists
            logger.info("  " * indent + f"{key}: [{len(value)} items]")
        else:
            logger.info("  " * indent + f"{key}: {value}")


def log_model_initialization(
    logger: logging.Logger,
    config,  # SSVAEConfig type
    architecture_code: str,
) -> None:
    """Log model initialization details.

    Provides a structured summary of the model architecture being trained.

    Args:
        logger: Logger instance
        config: SSVAE configuration object
        architecture_code: Generated architecture code

    Example output:
        ════════════════════════════════════════════════════════════════════════════════
                                  Model Initialization
        ════════════════════════════════════════════════════════════════════════════════
        Architecture Code: mix10-dir_tau_ca-het

        Prior Configuration:
          Type: mixture
          Components (K): 10
          Dirichlet α: 5.0
          Diversity weight: -0.10 (encourages diversity)

        Classifier Configuration:
          Type: τ-classifier (latent-only)
          Smoothing α: 1.0

        Decoder Configuration:
          Component-aware: ✓
          Heteroscedastic: ✓
          σ range: [0.05, 0.5]

        Training Configuration:
          Learning rate: 0.001
          Batch size: 128
          Max epochs: 100
          Early stopping patience: 20
    """
    log_section_header(logger, "Model Initialization")

    logger.info(f"Architecture Code: {architecture_code}")
    logger.info("")

    # Prior configuration
    logger.info("Prior Configuration:")
    logger.info(f"  Type: {config.prior_type}")

    if config.is_mixture_based_prior():
        logger.info(f"  Components (K): {config.num_components}")

        if config.prior_type == "mixture":
            if config.dirichlet_alpha is not None and config.dirichlet_alpha > 0:
                logger.info(f"  Dirichlet α: {config.dirichlet_alpha}")
            if config.component_diversity_weight != 0:
                direction = (
                    "encourages diversity"
                    if config.component_diversity_weight < 0
                    else "discourages diversity"
                )
                logger.info(
                    f"  Diversity weight: {config.component_diversity_weight} ({direction})"
                )
            if config.learnable_pi:
                logger.info("  Learnable π: ✓")

        elif config.prior_type == "vamp":
            logger.info(f"  Initialization: {config.vamp_pseudo_init_method}")
            logger.info(f"  MC samples (KL): {config.vamp_num_samples_kl}")
            logger.info(f"  Pseudo LR scale: {config.vamp_pseudo_lr_scale}")

        elif config.prior_type == "geometric_mog":
            logger.info(f"  Arrangement: {config.geometric_arrangement}")
            logger.info(f"  Radius: {config.geometric_radius}")
            logger.info("  ⚠ WARNING: Induces topology (diagnostic only)")

    logger.info("")

    # Classifier configuration
    logger.info("Classifier Configuration:")
    if config.use_tau_classifier:
        logger.info("  Type: τ-classifier (latent-only)")
        logger.info(f"  Smoothing α: {config.tau_smoothing_alpha}")
    else:
        logger.info("  Type: Standard head")
    logger.info("")

    # Decoder configuration
    logger.info("Decoder Configuration:")
    logger.info(
        f"  Component-aware: {'✓' if config.use_component_aware_decoder else '✗'}"
    )
    logger.info(
        f"  Heteroscedastic: {'✓' if config.use_heteroscedastic_decoder else '✗'}"
    )
    if config.use_heteroscedastic_decoder:
        logger.info(f"  σ range: [{config.sigma_min}, {config.sigma_max}]")
    if config.use_contrastive:
        logger.info(f"  Contrastive: ✓ (weight: {config.contrastive_weight})")
    logger.info("")

    # Training configuration
    logger.info("Training Configuration:")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Max epochs: {config.max_epochs}")
    logger.info(f"  Early stopping patience: {config.patience}")
    if config.kl_c_anneal_epochs > 0:
        logger.info(f"  KL_c annealing: {config.kl_c_anneal_epochs} epochs")
    logger.info("")


def log_training_epoch(
    logger: logging.Logger,
    epoch: int,
    max_epochs: int,
    losses: dict,
    metrics: Optional[dict] = None,
) -> None:
    """Log training progress for one epoch.

    Args:
        logger: Logger instance
        epoch: Current epoch number (1-indexed)
        max_epochs: Total number of epochs
        losses: Dictionary of loss values
        metrics: Optional dictionary of metric values

    Example:
        >>> log_training_epoch(
        ...     logger, epoch=5, max_epochs=100,
        ...     losses={"total": 198.3, "recon": 185.2, "kl_z": 12.1},
        ...     metrics={"accuracy": 0.87}
        ... )
        Epoch   5/100 | loss.total=198.30 | loss.recon=185.20 | loss.kl_z=12.10 | acc=0.870
    """
    # Build message components
    epoch_str = f"Epoch {epoch:3d}/{max_epochs}"

    # Format losses
    loss_strs = []
    for key, value in losses.items():
        if isinstance(value, float):
            loss_strs.append(f"{key}={value:.2f}")
        else:
            loss_strs.append(f"{key}={value}")

    # Format metrics
    metric_strs = []
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, float):
                metric_strs.append(f"{key}={value:.3f}")
            else:
                metric_strs.append(f"{key}={value}")

    # Combine into message
    message_parts = [epoch_str] + loss_strs + metric_strs
    message = " | ".join(message_parts)

    logger.info(message)
