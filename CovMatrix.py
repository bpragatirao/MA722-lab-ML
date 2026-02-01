import numpy as np

def covariance_matrix(data, ddof=1):
    X = np.asarray(data, dtype=float)
    if X.ndim != 2:
        print("data must be 2D (N samples x D features).")
    # center
    N = X.shape[0]
    if N <= ddof:
        print("Not enough samples for chosen ddof.")
    Xc = X - X.mean(axis=0)
    cov = (Xc.T @ Xc) / (N - ddof)
    return cov

# Example with 3 features, 4 samples
data = [
    [2.0, 1.0, 0.5],
    [3.0, 1.5, 0.7],
    [3.2, 1.8, 0.9],
    [2.8, 1.6, 0.8]
]
C = covariance_matrix(data, ddof=1)
print("Covariance matrix:\n", C)
