from pathlib import Path
import sys
import argparse

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import sklearn.manifold
from data.mnist import load_mnist_splits

from ssvae import SSVAE

DEFAULT_WEIGHTS = ROOT_DIR / "artifacts" / "checkpoints" / "ssvae.ckpt"
DEFAULT_OUTPUT = ROOT_DIR / "data" / "output_latent.npz"

(x_train_scaled, y_train), (x_test_scaled, y_test) = load_mnist_splits(normalize=True, reshape=True)

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
