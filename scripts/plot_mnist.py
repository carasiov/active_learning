import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


def plot_samples(images, labels, num_samples=10):
    plt.figure(figsize=(10, 2))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X.reshape(-1, 28, 28)
    plot_samples(X, y)
