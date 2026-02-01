from collections import defaultdict

def dense_to_sparse(mat):
    sparse = {}
    for i, row in enumerate(mat):
        for j, val in enumerate(row):
            if val != 0:
                sparse[(i, j)] = val
    return sparse

def sparse_matmul(A_sparse, A_shape, B_sparse, B_shape):
    m, p1 = A_shape
    p2, n = B_shape
    if p1 != p2:
        print("Inner dimensions must agree.")
    # Build index by column of A (k -> list of (i, Aik))
    A_by_k = defaultdict(list)
    for (i, k), val in A_sparse.items():
        A_by_k[k].append((i, val))
    # Build index by row of B (k -> list of (j, Bkj))
    B_by_k = defaultdict(list)
    for (k, j), val in B_sparse.items():
        B_by_k[k].append((j, val))

    C = defaultdict(float)
    # iterate only over k where both have nonzeros
    for k in set(A_by_k.keys()).intersection(B_by_k.keys()):
        for i, a_val in A_by_k[k]:
            for j, b_val in B_by_k[k]:
                C[(i, j)] += a_val * b_val
    # remove zeros (tolerance)
    tol = 1e-12
    C_clean = {k: v for k, v in C.items() if abs(v) > tol}
    return C_clean

# Example
A = [
    [1, 0, 2],
    [0, 0, 3]
]  # 2x3
B = [
    [0, 4],
    [0, 0],
    [5, 6]
]  # 3x2

A_sp = dense_to_sparse(A)
B_sp = dense_to_sparse(B)
C_sp = sparse_matmul(A_sp, (2,3), B_sp, (3,2))
print("Sparse result (i,j): value")
for k, v in sorted(C_sp.items()):
    print(k, ":", v)
