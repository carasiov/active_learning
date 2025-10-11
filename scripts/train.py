import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from configs.base import SSVAEConfig, get_architecture_defaults

DEFAULT_LABELS = BASE_DIR / "data" / "labels.csv"
DEFAULT_WEIGHTS = BASE_DIR / "artifacts" / "checkpoints" / "ssvae.ckpt"


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
    parser.add_argument("--xla-flags", type=str, default=None, help="Override XLA_FLAGS value")
    return parser.parse_args()


def load_training_images() -> np.ndarray:
    data, _ = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    data = data.astype(np.float32) / 255.0
    train_flat = data[:60000]
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_flat)
    return train_scaled.reshape(-1, 28, 28)


def load_label_array(path: Path, num_samples: int) -> np.ndarray:
    user_labels = pd.read_csv(path, header=0, index_col=0).dropna()
    index = pd.DataFrame(index=np.arange(num_samples))
    return index.join(user_labels).values


def build_config(args: argparse.Namespace) -> SSVAEConfig:
    base = SSVAEConfig()
    encoder_type = args.encoder_type or base.encoder_type
    decoder_type = args.decoder_type or base.decoder_type
    defaults = get_architecture_defaults(encoder_type)
    return SSVAEConfig(
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        latent_dim=args.latent_dim or base.latent_dim,
        batch_size=args.batch_size or defaults["batch_size"],
        max_epochs=args.max_epochs or base.max_epochs,
        patience=args.patience or base.patience,
        learning_rate=args.learning_rate or defaults["learning_rate"],
        recon_weight=args.recon_weight or defaults["recon_weight"],
        kl_weight=args.kl_weight or base.kl_weight,
        label_weight=args.label_weight or defaults["label_weight"],
        xla_flags=args.xla_flags or defaults.get("xla_flags"),
    )


def main() -> None:
    args = parse_args()
    train_images = load_training_images()
    labels = load_label_array(Path(args.labels), train_images.shape[0])
    config = build_config(args)

    print(
        f"Creating SSVAE with encoder={config.encoder_type}, decoder={config.decoder_type}, latent_dim={config.latent_dim}",
        flush=True,
    )
    print(
        f"  batch_size={config.batch_size}, lr={config.learning_rate}, recon_weight={config.recon_weight}, "
        f"kl_weight={config.kl_weight}, label_weight={config.label_weight}",
        flush=True,
    )

    if config.xla_flags:
        os.environ["XLA_FLAGS"] = config.xla_flags

    from ssvae import SSVAE  # noqa: E402  # Import after setting XLA flags

    vae = SSVAE(input_dim=(28, 28), config=config)
    vae.fit(train_images, labels, weights_path=args.weights)


if __name__ == "__main__":
    main()
