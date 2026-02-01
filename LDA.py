import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def lda():
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data  # (150,4)
    y = iris.target
    class_labels = np.unique(y)
    n_features = X.shape[1]
    overall_mean = X.mean(axis=0)

    # SW: within-class scatter
    SW = np.zeros((n_features, n_features))
    SB = np.zeros((n_features, n_features))
    for cl in class_labels:
        Xc = X[y == cl]
        mean_cl = Xc.mean(axis=0)
        # within-class scatter: sum over (x - mean_cl)(x - mean_cl)^T
        Xc_centered = Xc - mean_cl
        SW += Xc_centered.T @ Xc_centered
        n_cl = Xc.shape[0]
        mean_diff = (mean_cl - overall_mean).reshape(-1,1)
        SB += n_cl * (mean_diff @ mean_diff.T)

    # Solve eigenproblem for SW^-1 SB
    # For numerical stability, use pseudo-inverse if SW is singular
    try:
        SW_inv = np.linalg.inv(SW)
    except np.linalg.LinAlgError:
        SW_inv = np.linalg.pinv(SW)

    M = SW_inv @ SB
    eigvals, eigvecs = np.linalg.eig(M)
    # sort eigenvectors by descending eigenvalue (real parts)
    idx = np.argsort(np.real(eigvals))[::-1]
    eigvals = np.real(eigvals[idx])
    eigvecs = np.real(eigvecs[:, idx])

    # Choose top 2 eigenvectors (discriminant components)
    W = eigvecs[:, :2]  # projection matrix
    X_proj = (X - overall_mean) @ W  # center then project

    # Plot
    plt.figure(figsize=(8,6))
    colors = ['navy', 'turquoise', 'darkorange']
    target_names = iris.target_names
    for color, i, target_name in zip(colors, [0,1,2], target_names):
        plt.scatter(X_proj[y == i, 0], X_proj[y == i, 1], alpha=.8, color=color, label=target_name, s=60, edgecolor='k')
    plt.xlabel("LD1")
    plt.ylabel("LD2")
    plt.legend()
    plt.title("LDA projection (top 2 discriminants) on Iris")
    plt.grid(alpha=0.3)
    plt.show()

    print("\nLDA results:")
    print("Eigenvalues (top):", eigvals[:4])
    print("Projection matrix W shape:", W.shape)
    return SW, SB, eigvals, W, X_proj

# Offer LDA demo
do_lda = input("\nRun LDA demo on Iris dataset? (Y/n): ").strip().lower()
if do_lda in ("", "y", "yes"):
    SW, SB, eigvals, W, X_proj = lda()
    print("Finished LDA demo.")