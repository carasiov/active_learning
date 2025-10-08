from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.datasets import mnist

#from src_tf.vae_model_tf import SSVAE, SSCVAE
from src_tf.vae_model_jax import SSVAE, SSCVAE
PATH_LABELS = BASE_DIR / "data" / "labels.csv"
WEIGHTS_PATH = BASE_DIR / "models" / "ssvae_test_ja10.weights.h5"

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

user_labels = pd.read_csv(PATH_LABELS, header=0, index_col=0).dropna()

labels = pd.DataFrame(x_train_scaled.reshape(-1,28*28))[[]].join(user_labels).values

vae = SSVAE(input_dim=(28,28))
# vae.load_model_weights(weights_path=str(BASE_DIR / "models" / "ssvae_2d.weights.h5"))
history = vae.fit(x_train_scaled, labels, weights_path=str(WEIGHTS_PATH))
