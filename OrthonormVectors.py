import math

def dot_pdt(v1, v2):
    # Dot product of Vectors
    return sum(x * y for x, y in zip(v1, v2))

def magnitude(v):
    # Euclidean Distance Calculation
    return math.sqrt(sum(x ** 2 for x in v))

def is_orthonormal(vectors, tol=1e-9):
    n = len(vectors)
    if n == 0:
        print("No vectors provided.")

    dim = len(vectors[0])
    # Check all vectors have same dimension
    for v in vectors:
        if len(v) != dim:
            print("All vectors must have the same dimension.")

    # Check unit length
    for i, v in enumerate(vectors):
        mag = magnitude(v)
        if not math.isclose(mag, 1.0, rel_tol=tol, abs_tol=tol):
            print(f"Vector {i+1} is not unit length (‖v‖ = {mag:.6f}).")
            return False

    # Check orthogonality
    for i in range(n):
        for j in range(i + 1, n):
            dot = dot_pdt(vectors[i], vectors[j])
            if not math.isclose(dot, 0.0, abs_tol=tol):
                print(f"Vectors {i+1} and {j+1} are not orthogonal (dot = {dot:.6f}).")
                return False

    return True


print("Check if a set of vectors is orthonormal.")
D = int(input("Enter number of vectors: ").strip())
vectors = []
for i in range(D):
    vec = list(map(float, input(f"Enter vector {i+1} components (space-separated): ").split()))
    vectors.append(vec)

if is_orthonormal(vectors):
    print("\n The given set of vectors IS orthonormal.")
else:
    print("\n The given set of vectors is NOT orthonormal.")

