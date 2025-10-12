import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.base import SSVAEConfig
from data.mnist import load_train_images_for_ssvae

DEFAULT_LABELS = ROOT_DIR / "data" / "mnist" / "labels.csv"
DEFAULT_WEIGHTS = ROOT_DIR / "artifacts" / "checkpoints" / "ssvae.ckpt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SSVAE on MNIST")
    parser.add_argument("--labels", type=str, default=str(DEFAULT_LABELS), help="Path to labels.csv")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS), help="Output weights path")
    parser.add_argument("--encoder-type", type=str, choices=["dense", "conv"], default=None, help="Override encoder type")
    parser.add_argument("--decoder-type", type=str, choices=["dense", "conv"], default=None, help="Override decoder type")
    parser.add_argument("--latent-dim", type=int, default=None, help="Override latent dimensionality")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override maximum training epochs")
    parser.add_argument("--patience", type=int, default=None, help="Override early stopping patience")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override optimizer learning rate")
    parser.add_argument("--recon-weight", type=float, default=None, help="Override reconstruction loss weight")
    parser.add_argument("--kl-weight", type=float, default=None, help="Override KL loss weight")
    parser.add_argument("--label-weight", type=float, default=None, help="Override classification loss weight")
    parser.add_argument("--weight-decay", type=float, default=None, help="Override weight decay")
    parser.add_argument("--dropout-rate", type=float, default=None, help="Override dropout rate in classifier")
    parser.add_argument(
        "--monitor-metric",
        type=str,
        default=None,
        choices=["auto", "loss", "classification_loss"],
        help="Metric to monitor for early stopping",
    )
    parser.add_argument("--xla-flags", type=str, default=None, help="Override XLA_FLAGS value")
    return parser.parse_args()


def load_training_images() -> np.ndarray:
    return load_train_images_for_ssvae()


def load_label_array(path: Path, num_samples: int) -> np.ndarray:
    user_labels = pd.read_csv(path, header=0, index_col=0).dropna()
    index = pd.DataFrame(index=np.arange(num_samples))
    return index.join(user_labels).values


def build_config(args: argparse.Namespace) -> SSVAEConfig:
    config_kwargs = {}
    if args.encoder_type is not None:
        config_kwargs["encoder_type"] = args.encoder_type
    if args.decoder_type is not None:
        config_kwargs["decoder_type"] = args.decoder_type
    if args.latent_dim is not None:
        config_kwargs["latent_dim"] = args.latent_dim
    if args.batch_size is not None:
        config_kwargs["batch_size"] = args.batch_size
    if args.max_epochs is not None:
        config_kwargs["max_epochs"] = args.max_epochs
    if args.patience is not None:
        config_kwargs["patience"] = args.patience
    if args.learning_rate is not None:
        config_kwargs["learning_rate"] = args.learning_rate
    if args.recon_weight is not None:
        config_kwargs["recon_weight"] = args.recon_weight
    if args.kl_weight is not None:
        config_kwargs["kl_weight"] = args.kl_weight
    if args.label_weight is not None:
        config_kwargs["label_weight"] = args.label_weight
    if args.weight_decay is not None:
        config_kwargs["weight_decay"] = args.weight_decay
    if args.dropout_rate is not None:
        config_kwargs["dropout_rate"] = args.dropout_rate
    if args.monitor_metric is not None:
        config_kwargs["monitor_metric"] = args.monitor_metric
    if args.xla_flags is not None:
        config_kwargs["xla_flags"] = args.xla_flags
    return SSVAEConfig(**config_kwargs)


def main() -> None:
    args = parse_args()
    train_images = load_training_images()
    labels = load_label_array(Path(args.labels), train_images.shape[0])
    config = build_config(args)

    if config.xla_flags:
        os.environ["XLA_FLAGS"] = config.xla_flags

    from ssvae import SSVAE  # noqa: E402  # Import after adjusting sys.path

    vae = SSVAE(input_dim=(28, 28), config=config)
    vae.fit(train_images, labels, weights_path=args.weights)


if __name__ == "__main__":
    main()
