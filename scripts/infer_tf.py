from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import sklearn.manifold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.datasets import mnist

from src_tf.vae_model_jax import SSVAE, SSCVAE
WEIGHTS_PATH = BASE_DIR / "models" / "ssvae_test_jax8.weights.h5"
OUTPUT_PATH = BASE_DIR / "data" / "output_jax_2.npz"

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Bilder auf float umstellen
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# Bilder flatten (28x28 -> 784)
x_train_flat = x_train.reshape((x_train.shape[0], -1))  # shape: (60000, 784)
x_test_flat = x_test.reshape((x_test.shape[0], -1))     # shape: (10000, 784)

# StandardScaler anwenden
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train_flat)
x_test_scaled = scaler.transform(x_test_flat)

# Optional: zurück in Bildform, falls gebraucht
x_train_scaled = x_train_scaled.reshape((-1, 28, 28))
x_test_scaled = x_test_scaled.reshape((-1, 28, 28))


vae = SSVAE(input_dim=(28, 28))
vae.load_model_weights(weights_path=str(WEIGHTS_PATH))
latent, reconstruted, pred_classes, pred_certainty = vae.predict(x_train_scaled)

# # t-SNE embedding of the latent space
# tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
# tsne_output = tsne.fit_transform(latent)
tsne_output=latent

np.savez_compressed(OUTPUT_PATH,
                    input=x_train_scaled,
                    tsne=tsne_output,
                    labels=y_train,
                    latent=latent,
                    reconstruted=reconstruted,
                    pred_classes=pred_classes,
                    pred_certainty=pred_certainty
                    )
