import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# MNIST-Datensatz laden
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Einige Samples anzeigen
def plot_samples(images, labels, num_samples=10):
    plt.figure(figsize=(10, 2))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Zeige die ersten 10 Samples aus dem Trainingsdatensatz
plot_samples(x_train, y_train)
