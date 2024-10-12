import sympy as sp

def commutator(matrix_a, matrix_b):
    A = sp.Matrix(matrix_a) if not isinstance(matrix_a, sp.Matrix) else matrix_a
    B = sp.Matrix(matrix_b) if not isinstance(matrix_b, sp.Matrix) else matrix_b

    if A.shape != B.shape:
        raise ValueError("Both matrices must be of the same dimensions.")
    if A.rows != A.cols:
        raise ValueError("Matrices must be square to compute the commutator.")

    AB = A * B
    BA = B * A

    comm = AB - BA
    return comm


if __name__ == "__main__":
    # Define two 2x2 matrices
    matrix1 = [[0, 1],
               [-1, 0]]

    matrix2 = [[2, 3],
               [4, 5]]

    # Compute the commutator
    comm = commutator(matrix1, matrix2)

    print("Matrix A:")
    sp.pprint(sp.Matrix(matrix1))

    print("\nMatrix B:")
    sp.pprint(sp.Matrix(matrix2))

    print("\nCommutator [A, B]:")
    sp.pprint(comm)
