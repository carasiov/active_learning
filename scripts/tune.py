import argparse
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from configs.base import SSVAEConfig
from scripts.train import load_label_array, load_training_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for SSVAE (random search)")
    parser.add_argument("--labels", type=str, default=str(BASE_DIR / "data" / "labels.csv"))
    parser.add_argument("--trials", type=int, default=10, help="Number of random trials")
    parser.add_argument("--max-epochs", type=int, default=60, help="Max epochs per trial")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience per trial")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--weights-dir", type=str, default=str(BASE_DIR / "artifacts" / "tuning"), help="Directory to store trial checkpoints")
    parser.add_argument("--encoder-type", type=str, default=None, choices=["dense", "conv"], help="Override encoder type for all trials")
    parser.add_argument("--decoder-type", type=str, default=None, choices=["dense", "conv"], help="Override decoder type for all trials")
    return parser.parse_args()


def sample_trial_cfg(default: SSVAEConfig) -> Dict[str, object]:
    # Discrete search spaces (kept small for practicality)
    lrs = [1e-3, 5e-4, 3e-4, 1e-4]
    wds = [0.0, 1e-4, 5e-4, 1e-3]
    drops = [0.0, 0.1, 0.2, 0.4]
    lbs = [1.0, 2.0, 5.0, 10.0, 25.0, 50.0]
    bsz = [512, 1024, 2048, default.batch_size]

    cfg = {
        "learning_rate": random.choice(lrs),
        "weight_decay": random.choice(wds),
        "dropout_rate": random.choice(drops),
        "label_weight": random.choice(lbs),
        "batch_size": random.choice(bsz),
    }
    return cfg


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    images = load_training_images()
    labels = load_label_array(Path(args.labels), images.shape[0])

    out_dir = Path(args.weights_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "tuning_results.csv"

    # Header for CSV
    with open(results_csv, "w", encoding="utf-8") as f:
        f.write(
            "trial,learning_rate,weight_decay,dropout_rate,label_weight,batch_size,best_val_cls_epoch,best_val_cls\n"
        )

    best_overall = None
    best_trial_idx = -1

    for t in range(args.trials):
        base_cfg = SSVAEConfig()
        if args.encoder_type is not None:
            base_cfg.encoder_type = args.encoder_type
        if args.decoder_type is not None:
            base_cfg.decoder_type = args.decoder_type
        base_cfg.max_epochs = args.max_epochs
        base_cfg.patience = args.patience

        trial_cfg = sample_trial_cfg(base_cfg)
        for k, v in trial_cfg.items():
            setattr(base_cfg, k, v)

        # Build checkpoint path per trial
        ckpt_path = out_dir / f"trial_{t+1:03d}.ckpt"

        # Defer import to after XLA flag usage if needed
        from ssvae import SSVAE  # noqa: E402

        model = SSVAE(input_dim=(28, 28), config=base_cfg)
        history = model.fit(images, labels, weights_path=str(ckpt_path))

        val_cls = history["val_classification_loss"]
        best_val = float(np.min(val_cls)) if len(val_cls) else float("inf")
        best_epoch = int(np.argmin(val_cls) + 1) if len(val_cls) else -1

        with open(results_csv, "a", encoding="utf-8") as f:
            f.write(
                f"{t+1},{trial_cfg['learning_rate']},{trial_cfg['weight_decay']},{trial_cfg['dropout_rate']},{trial_cfg['label_weight']},{trial_cfg['batch_size']},{best_epoch},{best_val:.6f}\n"
            )

        print(
            f"Trial {t+1:03d} | lr={trial_cfg['learning_rate']} wd={trial_cfg['weight_decay']} drop={trial_cfg['dropout_rate']} "
            f"label_w={trial_cfg['label_weight']} bs={trial_cfg['batch_size']} -> best val_cls {best_val:.4f} @ epoch {best_epoch}"
        )

        if best_overall is None or best_val < best_overall:
            best_overall = best_val
            best_trial_idx = t + 1

    print(
        f"Best trial: {best_trial_idx:03d} with best val_classification_loss={best_overall:.4f}. "
        f"See {results_csv} for details."
    )


if __name__ == "__main__":
    main()

