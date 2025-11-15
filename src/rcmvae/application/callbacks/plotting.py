"""Plotting callbacks for training visualization."""

from __future__ import annotations

from pathlib import Path

from .base import HistoryDict, TrainingCallback

try:  # optional dependency
    import matplotlib.pyplot as plt

    _HAS_PLT = True
except Exception:  # pragma: no cover - optional at runtime
    plt = None  # type: ignore
    _HAS_PLT = False


class LossCurvePlotter(TrainingCallback):
    """Creates loss curve plots at the end of training."""

    def __init__(self, save_path: str | Path):
        self.save_path = Path(save_path)

    def on_train_end(self, history: HistoryDict, trainer: "Trainer") -> None:
        if not _HAS_PLT:
            return

        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))  # type: ignore[arg-type]

        axes[0].plot(history["loss"], label="Training Loss")
        axes[0].plot(history["val_loss"], label="Validation Loss")
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss and Validation Loss")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(history["reconstruction_loss"], label="Reconstruction", color="blue")
        axes[1].plot(history["val_reconstruction_loss"], label="Val Reconstruction", color="cyan")
        axes[1].plot(history["kl_loss"], label="KL", color="red")
        axes[1].plot(history["val_kl_loss"], label="Val KL", color="orange")
        axes[1].plot(history["classification_loss"], label="Classification", color="green")
        axes[1].plot(history["val_classification_loss"], label="Val Classification", color="lime")
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Component Losses")
        axes[1].legend()
        axes[1].grid(True)

        fig.tight_layout()
        fig.savefig(self.save_path)
        plt.close(fig)  # type: ignore[arg-type]
