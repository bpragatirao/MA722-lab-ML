import numpy as np

def svd(A):
    A = np.asarray(A, dtype=float)
    m, n = A.shape

    U, s, VT = np.linalg.svd(A, full_matrices=True)
    return U, s, VT

# Example
A = np.array([[3.0, 1.0], [1.0, 3.0], [0.0, 2.0]])  # 3x2
U, S, VT = svd(A)
print("U shape:", U.shape)
print("Singular values S:", S)
print("VT shape:", VT.shape)
# Check reconstruction with diag(S)
Sigma = np.zeros((U.shape[0], VT.shape[0]))
k = min(A.shape)
Sigma[:k, :k] = np.diag(S)
A_rec = U @ Sigma @ VT
print("Reconstruction error (Frobenius):", np.linalg.norm(A - A_rec))
