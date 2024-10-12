from sympy import Matrix, I, exp, pi, sqrt
from sympy import symbols


def Identity():
    """Identity I gate"""
    return Matrix([[1, 0],
                  [0, 1]])

def H():
    """Hadamard gate"""
    return (1/sqrt(2)) * Matrix([[1, 1],
                                [1, -1]])

def Y():
    """Pauli Y gate"""
    return Matrix([[0, -I],
                  [I, 0]])

def Z():
    """Pauli Z gate"""
    return Matrix([[1, 0],
                  [0, -1]])

def T():
    """T gate"""
    return Matrix([[1, 0],
                  [0, exp(I * pi / 4)]])

def X():
    """Pauli X gate"""
    return Matrix([[0, 1],
                  [1, 0]])

def CX():
    """Controlled X (CNOT) gate"""
    return Matrix([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0]])

def CU(U):
    """
    Controlled U gate.
    
    Parameters:
    U (Matrix): A 2x2 unitary matrix.
    
    Returns:
    Matrix: The 4x4 controlled U matrix.
    """
    return Matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, U[0,0], U[0,1]],
        [0, 0, U[1,0], U[1,1]]
    ])

def CNOT():
    """Controlled NOT gate (same as CX)"""
    return CX()

def CCNOT():
    """Toffoli gate (Controlled Controlled NOT)"""
    return Matrix([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0]
    ])

def SWAP():
    """SWAP gate"""
    return Matrix([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])

def P(phi):
    """
    Phase gate P(phi).
    
    Parameters:
    phi (symbol or number): Phase angle in radians.
    
    Returns:
    Matrix: The 2x2 phase gate matrix.
    """
    return Matrix([
        [1, 0],
        [0, exp(I * phi)]
    ])


if __name__ == "__main__":
    theta = symbols('theta')
    U = Matrix([
        [exp(I * theta / 2), 0],
        [0, exp(-I * theta / 2)]
    ])

    print("Hadamard gate H:")
    print(H())