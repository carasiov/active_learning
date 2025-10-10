from pathlib import Path
import sys
import argparse

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_openml

from ssvae import SSVAE, SSCVAE
from configs.base import SSVAEConfig

DEFAULT_LABELS = BASE_DIR / "data" / "labels.csv"
DEFAULT_WEIGHTS = BASE_DIR / "artifacts" / "checkpoints" / "ssvae.ckpt"

X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X.astype(np.float32) / 255.0
y = y.astype(np.int32)
x_train_flat, x_test_flat = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# StandardScaler anwenden
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train_flat)
x_test_scaled = scaler.transform(x_test_flat)

X_SHAPE = (-1, 28, 28)
x_train_scaled = x_train_scaled.reshape(X_SHAPE)
x_test_scaled = x_test_scaled.reshape(X_SHAPE)

parser = argparse.ArgumentParser(description="Train SSVAE on MNIST")
parser.add_argument("--labels", type=str, default=str(DEFAULT_LABELS), help="Path to labels.csv")
parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS), help="Output weights path")
parser.add_argument("--encoder-type", type=str, default="dense", choices=["dense", "conv"], help="Encoder type")
parser.add_argument("--decoder-type", type=str, default="dense", choices=["dense", "conv"], help="Decoder type")
parser.add_argument("--latent-dim", type=int, default=2, help="Latent dimensionality")
parser.add_argument("--batch-size", type=int, default=4 * 1024, help="Batch size")
parser.add_argument("--max-epochs", type=int, default=200, help="Max training epochs")
parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
args = parser.parse_args()

user_labels = pd.read_csv(args.labels, header=0, index_col=0).dropna()

labels = pd.DataFrame(x_train_scaled.reshape(-1,28*28))[[]].join(user_labels).values

config = SSVAEConfig(
    encoder_type=args.encoder_type,
    decoder_type=args.decoder_type,
    latent_dim=args.latent_dim,
    batch_size=args.batch_size,
    max_epochs=args.max_epochs,
    patience=args.patience,
    learning_rate=args.learning_rate,
)

print(
    f"Creating SSVAE with encoder={config.encoder_type}, decoder={config.decoder_type}, latent_dim={config.latent_dim}",
    flush=True,
)
vae = SSVAE(input_dim=(28, 28), config=config)
history = vae.fit(x_train_scaled, labels, weights_path=args.weights)
