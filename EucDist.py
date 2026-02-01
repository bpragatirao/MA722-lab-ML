import math
import numpy as np

def euc_dist(v, w):
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)
    if v.shape != w.shape:
        print("Vectors must have same shape.")
    return float(np.linalg.norm(v - w))

# Example
v = [1.0, 2.0, 3.0]
w = [2.0, 0.0, 4.0]
print("Distance:", euc_dist(v, w))
