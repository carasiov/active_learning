"""Logging callbacks for training metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from utils import get_device_info

from .base_callback import HistoryDict, MetricsDict, TrainingCallback


class ConsoleLogger(TrainingCallback):
    """Prints formatted metric tables to the console."""

    def __init__(self) -> None:
        self._header_printed = False

    def on_train_start(self, trainer: "Trainer") -> None:
        self._header_printed = False

        device_type, device_count = get_device_info()
        if device_type:
            plural = "s" if device_count != 1 else ""
            print(
                f"Training on {device_type.upper()} ({device_count} device{plural})",
                flush=True,
            )

    def on_epoch_end(
        self,
        epoch: int,
        metrics: Dict[str, MetricsDict],
        history: HistoryDict,
        trainer: "Trainer",
    ) -> None:
        train_metrics = metrics["train"]
        val_metrics = metrics["val"]

        metric_columns = [
            ("Train.loss", train_metrics, "loss"),
            ("Val.loss", val_metrics, "loss"),
            ("Train.loss_np", train_metrics, "loss_no_global_priors"),
            ("Val.loss_np", val_metrics, "loss_no_global_priors"),
            ("Train.rec", train_metrics, "reconstruction_loss"),
            ("Val.rec", val_metrics, "reconstruction_loss"),
            ("Train.kl", train_metrics, "kl_loss"),
            ("Val.kl", val_metrics, "kl_loss"),
            ("Train.cls", train_metrics, "classification_loss"),
            ("Val.cls", val_metrics, "classification_loss"),
        ]

        if "contrastive_loss" in train_metrics and "contrastive_loss" in val_metrics:
            metric_columns.extend(
                [
                    ("Train.con", train_metrics, "contrastive_loss"),
                    ("Val.con", val_metrics, "contrastive_loss"),
                ]
            )

        header_parts = [f"{'Epoch':>5}"]
        row_parts = [f"{epoch + 1:>5d}"]

        for label, source, key in metric_columns:
            header_parts.append(f"{label:>12}")
            row_parts.append(f"{float(source[key]):>12.4f}")

        if not self._header_printed and epoch == 0:
            header_line = " | ".join(header_parts)
            divider = "-" * len(header_line)
            print(header_line, flush=True)
            print(divider, flush=True)
            self._header_printed = True

        print(" | ".join(row_parts), flush=True)


class CSVExporter(TrainingCallback):
    """Exports training history to a CSV file."""

    def __init__(self, save_path: str | Path):
        self.save_path = Path(save_path)

    def on_train_end(self, history: HistoryDict, trainer: "Trainer") -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        headers = [
            "epoch",
            "loss",
            "val_loss",
            "reconstruction_loss",
            "val_reconstruction_loss",
            "kl_loss",
            "val_kl_loss",
            "classification_loss",
            "val_classification_loss",
        ]
        epochs = len(history["loss"])
        with open(self.save_path, "w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            for i in range(epochs):
                row = [
                    str(i + 1),
                    f"{history['loss'][i]:.8f}",
                    f"{history['val_loss'][i]:.8f}",
                    f"{history['reconstruction_loss'][i]:.8f}",
                    f"{history['val_reconstruction_loss'][i]:.8f}",
                    f"{history['kl_loss'][i]:.8f}",
                    f"{history['val_kl_loss'][i]:.8f}",
                    f"{history['classification_loss'][i]:.8f}",
                    f"{history['val_classification_loss'][i]:.8f}",
                ]
                f.write(",".join(row) + "\n")
