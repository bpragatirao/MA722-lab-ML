import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

def iris_pca_plot():
    iris = load_iris()
    X = iris.data        # shape (150, 4)
    y = iris.target      # 0,1,2
    target_names = iris.target_names

    pca = PCA(n_components=2)
    X_r = pca.fit_transform(X)

    plt.figure(figsize=(8,6))
    colors = ['navy', 'turquoise', 'darkorange']
    for color, i, target_name in zip(colors, [0,1,2], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, lw=1, alpha=0.7, label=target_name)
    plt.legend()
    plt.title("PCA of Iris dataset (4 → 2)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.show()

    # Print explained variance
    print("Explained variance ratios:", pca.explained_variance_ratio_)
    print("Cumulative explained variance (PC1+PC2):", pca.explained_variance_ratio_.sum())

# Run
iris_pca_plot()
