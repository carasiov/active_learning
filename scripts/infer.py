from pathlib import Path
import sys
import argparse

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import sklearn.manifold
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_openml

from ssvae import SSVAE, SSCVAE

DEFAULT_WEIGHTS = BASE_DIR / "artifacts" / "checkpoints" / "ssvae.ckpt"
DEFAULT_OUTPUT = BASE_DIR / "data" / "output_latent.npz"

X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X.astype(np.float32) / 255.0
y = y.astype(np.int32)

# Reshape to image format (28x28)
X = X.reshape((-1, 28, 28))

# Split into train and test
x_train_scaled, x_test_scaled = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

parser = argparse.ArgumentParser(description="Infer latent space and save outputs")
parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS), help="Path to model weights")
parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output .npz path")
parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Dataset split")
args = parser.parse_args()

data_split = x_train_scaled if args.split == "train" else x_test_scaled
labels_split = y_train if args.split == "train" else y_test

vae = SSVAE(input_dim=(28, 28))
vae.load_model_weights(weights_path=args.weights)
latent, reconstruted, pred_classes, pred_certainty = vae.predict(data_split)

# # t-SNE embedding of the latent space
# tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
# tsne_output = tsne.fit_transform(latent)
tsne_output=latent

np.savez_compressed(args.output,
                    input=data_split,
                    tsne=tsne_output,
                    labels=labels_split,
                    latent=latent,
                    reconstruted=reconstruted,
                    pred_classes=pred_classes,
                    pred_certainty=pred_certainty
                    )
