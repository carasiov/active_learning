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
args = parser.parse_args()

user_labels = pd.read_csv(args.labels, header=0, index_col=0).dropna()

labels = pd.DataFrame(x_train_scaled.reshape(-1,28*28))[[]].join(user_labels).values

vae = SSVAE(input_dim=(28, 28))
history = vae.fit(x_train_scaled, labels, weights_path=args.weights)
