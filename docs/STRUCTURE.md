Project Structure Overview

Top-level layout
- configs/: Model/training configuration (`SSVAEConfig`)
- model_components/: Flax modules (encoders/decoders/classifier) and model factory
- training/: Losses, training loop, interactive trainer, train state
- scripts/: User-facing entry points (train, infer, view, utilities)
- ssvae/: Canonical import path for the JAX SSVAE backend
- artifacts/checkpoints/: Saved weights/checkpoints
- artifacts/progress/: Training history artifacts and plots
- ROOT/: Refactor specs and progress tracker (authoritative guidance)
- docs/: Documentation and assets
- data/: Dataset artifacts produced/consumed by scripts (e.g., labels, outputs)

Key modules
- configs/base.py: `SSVAEConfig` — hyperparameters, architecture toggles, loss flags
- model_components/encoders.py | decoders.py | classifier.py: Flax building blocks
- model_components/factory.py: Build model components and expose dims from config
- training/losses.py: Pure losses (MSE, KL, masked classification; contrastive stub)
- training/trainer.py: Epoch loop with early stopping + checkpointing
- training/interactive_trainer.py: Incremental training between labeling rounds
- ssvae/models.py: JAX SSVAE public API (canonical path)

Scripts
- scripts/train.py: Train on MNIST, reads `data/labels.csv`, saves weights/history (CLI flags `--labels`, `--weights`)
- scripts/infer.py: Run inference and persist latent/outputs (CLI flags `--weights`, `--output`, `--split`)
- scripts/view_latent.py: Interactive viewer to color/label points, writes `data/labels.csv`
- scripts/plot_mnist.py: Quick utility to visualize MNIST samples

Docs and specs
- ROOT/refactor_spec_part1.txt | part2.txt: Requirements and execution plan
- ROOT/progress_tracker.txt: Step-by-step progress log
- docs/: Place for architecture, usage, and design documents

Notes
- Prefer `from ssvae import SSVAE` in new code.
