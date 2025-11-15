"""Example demonstrating the experiment logging system.

This example shows:
1. Setting up structured logging with multiple handlers
2. Logging model initialization with architecture details
3. Logging training progress with metrics
4. Using section headers for visual organization
5. Different log levels (INFO, DEBUG, WARNING, ERROR)

Run this example:
    python use_cases/experiments/examples/logging_example.py

Output:
    - Clean terminal output (no timestamps)
    - Detailed logs/experiment.log
    - Filtered logs/training.log (training only)
    - logs/errors.log (warnings/errors only)
"""
from pathlib import Path
import tempfile

from rcmvae.domain.config import SSVAEConfig
from use_cases.experiments.src.naming import generate_architecture_code
from infrastructure.logging import (
    setup_experiment_logging,
    log_section_header,
    log_model_initialization,
    log_training_epoch,
)


def main():
    # Create temporary directory for logs
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "logs"

        print("=" * 80)
        print("Experiment Logging Example".center(80))
        print("=" * 80)
        print(f"\nLog directory: {log_dir}")
        print("\nOutput shows:")
        print("  - Clean terminal output (what you see now)")
        print("  - Detailed file logs (check log_dir after)")
        print("\n" + "=" * 80 + "\n")

        # ═════════════════════════════════════════════════════════════════════
        # 1. Setup logging
        # ═════════════════════════════════════════════════════════════════════

        logger = setup_experiment_logging(log_dir)

        # ═════════════════════════════════════════════════════════════════════
        # 2. Log experiment start
        # ═════════════════════════════════════════════════════════════════════

        log_section_header(logger, "Experiment Start")
        logger.info("Experiment: baseline_mixture")
        logger.info("Run ID: a3f9c2d1")
        logger.info("Timestamp: 20251112_143022")

        # ═════════════════════════════════════════════════════════════════════
        # 3. Log model initialization
        # ═════════════════════════════════════════════════════════════════════

        config = SSVAEConfig(
            prior_type="mixture",
            num_components=10,
            dirichlet_alpha=5.0,
            use_tau_classifier=True,
            use_component_aware_decoder=True,
            use_heteroscedastic_decoder=True,
            component_diversity_weight=-0.10,
            learnable_pi=True,
            max_epochs=100,
        )

        arch_code = generate_architecture_code(config)
        log_model_initialization(logger, config, arch_code)

        # ═════════════════════════════════════════════════════════════════════
        # 4. Log data configuration
        # ═════════════════════════════════════════════════════════════════════

        log_section_header(logger, "Data Configuration")
        logger.info("Dataset: MNIST")
        logger.info("Total samples: 10,000")
        logger.info("Labeled samples: 100 (1.0%)")
        logger.info("Validation split: 0.1")
        logger.info("")

        # ═════════════════════════════════════════════════════════════════════
        # 5. Log training progress
        # ═════════════════════════════════════════════════════════════════════

        log_section_header(logger, "Training Progress")

        # Simulate training epochs
        for epoch in range(1, 11):
            # Simulate decreasing loss
            total_loss = 200.0 - epoch * 2.0
            recon_loss = 180.0 - epoch * 1.8
            kl_z = 15.0 - epoch * 0.15
            kl_c = 5.0 - epoch * 0.05

            losses = {
                "loss.total": total_loss,
                "loss.recon": recon_loss,
                "loss.kl_z": kl_z,
                "loss.kl_c": kl_c,
            }

            # Simulate improving accuracy
            accuracy = 0.10 + epoch * 0.02

            metrics = {"acc": accuracy}

            log_training_epoch(logger, epoch, 100, losses, metrics)

        logger.info("")

        # ═════════════════════════════════════════════════════════════════════
        # 6. Log warnings and errors (these go to errors.log)
        # ═════════════════════════════════════════════════════════════════════

        logger.warning("Component 3 has low usage (< 1%)")
        logger.warning("Learning rate may be too high (loss oscillating)")

        # ═════════════════════════════════════════════════════════════════════
        # 7. Log debug info (only in experiment.log, not console)
        # ═════════════════════════════════════════════════════════════════════

        logger.debug("Gradient norm: 0.523")
        logger.debug("Parameter count: 1,234,567")
        logger.debug("Memory usage: 512 MB")

        # ═════════════════════════════════════════════════════════════════════
        # 8. Log completion
        # ═════════════════════════════════════════════════════════════════════

        log_section_header(logger, "Training Complete")
        logger.info("Total time: 123.45 seconds")
        logger.info("Best epoch: 87")
        logger.info("Final accuracy: 0.89")
        logger.info("")

        # ═════════════════════════════════════════════════════════════════════
        # Show log file contents
        # ═════════════════════════════════════════════════════════════════════

        print("\n" + "=" * 80)
        print("Log Files Created".center(80))
        print("=" * 80)

        for log_file in log_dir.glob("*.log"):
            print(f"\n{log_file.name}:")
            print("-" * 80)
            content = log_file.read_text()
            lines = content.split("\n")
            # Show first 10 and last 5 lines
            if len(lines) > 15:
                print("\n".join(lines[:10]))
                print(f"... ({len(lines) - 15} lines omitted) ...")
                print("\n".join(lines[-5:]))
            else:
                print(content)

        print("\n" + "=" * 80)
        print("Notice the differences:".center(80))
        print("=" * 80)
        print("  experiment.log: Full detail with timestamps")
        print("  training.log: Only epoch/loss messages")
        print("  errors.log: Only warnings and errors")
        print("  Terminal: Clean, no timestamps")


if __name__ == "__main__":
    main()
