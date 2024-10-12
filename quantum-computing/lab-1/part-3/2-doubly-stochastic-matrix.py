import sympy
from sympy import Matrix, Rational, zeros
import random
import itertools

def generate_doubly_stochastic_with_simple_fractions(N, num_terms=None):
    if num_terms is None:
        num_terms = N  # Default number of terms

    # Generate all permutations of size N
    permutations = list(itertools.permutations(range(N)))
    num_perms = len(permutations)

    # Randomly select a subset of permutation matrices
    selected_indices = random.sample(range(num_perms), num_terms)
    selected_perms = [permutations[i] for i in selected_indices]
    
    # Assign small integer weights to each selected permutation
    weights = [random.randint(1, 3) for _ in range(num_terms)]  # Weights between 1 and 3
    weight_sum = sum(weights)
    weights = [Rational(w, weight_sum) for w in weights]  # Normalize weights

    # Create the permutation matrices and form the convex combination
    A = zeros(N)
    for perm, w in zip(selected_perms, weights):
        P = zeros(N)
        for i in range(N):
            P[i, perm[i]] = 1
        A += w * P

    # Simplify fractions
    A = A.applyfunc(lambda x: x.simplify())
    return A


def is_doubly_stochastic(matrix, tol=1e-8):
    N, M = matrix.shape
    if N != M:
        print("Matrix is not square.")
        return False

    for i in range(N):
        for j in range(M):
            if matrix[i, j] < 0:
                print(f"Negative entry found at ({i}, {j}): {matrix[i,j]}")
                return False

    for i in range(N):
        row_sum = sum(matrix.row(i))
        row_sum_val = float(row_sum)
        if abs(row_sum_val - 1.0) > tol:
            print(f"Row {i} does not sum to 1. Sum = {row_sum_val}")
            return False

    for j in range(M):
        col_sum = sum(matrix.col(j))
        col_sum_val = float(col_sum)
        if abs(col_sum_val - 1.0) > tol:
            print(f"Column {j} does not sum to 1. Sum = {col_sum_val}")
            return False

    return True


def apply_matrix(M, v, k):
    if (k < 0):
        M = M.inv()
        k = k * -1
    for i in range(k):
        v = M @ v
    return v

N = 4
ds_matrix = generate_doubly_stochastic_with_simple_fractions(N)
print("Generated Doubly Stochastic Matrix:")
sympy.pprint(ds_matrix)

result = is_doubly_stochastic(ds_matrix)
print(f"Is the matrix doubly stochastic? {result}")


some_vec = sympy.Matrix([[70595635/282475249], [70612572/282475249], [70646454/282475249], [70620588/282475249]])

some_vec = apply_matrix(ds_matrix, some_vec, -10)

print(some_vec)