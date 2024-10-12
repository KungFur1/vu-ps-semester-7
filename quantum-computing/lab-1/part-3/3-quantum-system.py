import numpy as np
import sympy as sp
from sympy import I, Matrix, eye
import random
from sympy import Abs



def generate_random_phase_permutation_matrix(dim):
    """
    Generates a random phase-permutation unitary matrix of size dim x dim using SymPy.
    
    Parameters:
    dim (int): The dimension of the unitary matrix.
    
    Returns:
    sp.Matrix: A SymPy Matrix representing the unitary matrix.
    """
    # Step 1: Generate a random permutation of the rows/columns
    permutation = list(range(dim))
    random.shuffle(permutation)
    
    # Step 2: Create a permutation matrix
    P = Matrix.zeros(dim, dim)
    for i in range(dim):
        P[i, permutation[i]] = 1
    
    # Step 3: Generate random phase factors
    # For exactness, choose phase factors from {1, -1, I, -I}
    phase_options = [1, -1, I, -I]
    phases = [random.choice(phase_options) for _ in range(dim)]
    
    # Step 4: Create a diagonal phase matrix
    D = eye(dim)
    for i in range(dim):
        D[i, i] = phases[i]
    
    # Step 5: Multiply permutation matrix by phase matrix to get the unitary matrix
    U = P * D
    
    return U



def is_unitary(matrix):
    """
    Checks if a given SymPy matrix is unitary.

    Parameters:
    matrix (sp.Matrix): The matrix to check.

    Returns:
    bool: True if the matrix is unitary, False otherwise.
    """
    # Step 1: Check if the matrix is square
    if matrix.rows != matrix.cols:
        print("Matrix is not square.")
        return False
    
    dim = matrix.rows
    
    # Step 2: Compute the conjugate transpose (Hermitian adjoint) of the matrix
    matrix_dagger = matrix.conjugate().transpose()
    
    # Step 3: Compute the product U† * U
    product = matrix_dagger * matrix
    
    # Step 4: Generate the identity matrix of the same dimension
    identity = eye(dim)
    
    # Step 5: Check if U† * U equals the identity matrix
    if product.equals(identity):
        return True
    else:
        return False


def apply_matrix(M, v, k):
    if (k < 0):
        M = M.inv()
        k = k * -1
    for i in range(k):
        v = M @ v
    return v

def calculate_the_probability_vector(M, v, T):
    v = apply_matrix(M, v, T)
    probs = []
    for index, entry in enumerate(v):
        probs.append(Abs(entry)**2)
    
    return probs



if __name__ == "__main__":
    dim = 4
    unitary_matrix = generate_random_phase_permutation_matrix(dim)
    print("Random Phase-Permutation Unitary Matrix ({}x{}):".format(dim, dim))
    sp.pprint(unitary_matrix)
    print(is_unitary(unitary_matrix))
    vec0 = apply_matrix(unitary_matrix, sp.Matrix([1, 2, 3, 4]), 10)
    vec1 = apply_matrix(unitary_matrix, vec0, -10)

    print(vec0)
    print(vec1)

    # print(calculate_the_probability_vector(unitary_matrix, sp.Matrix([1, 0, 0, 0]), 2))

