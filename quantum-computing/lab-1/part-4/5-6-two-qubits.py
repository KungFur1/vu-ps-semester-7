import sympy
from sympy import kronecker_product, Matrix
from gates_3 import H, Y, Z, Identity, CX, SWAP


# HCY-HX
# HXZHHC
def full_system():
    return SWAP() @ CX() @ SWAP() @ Matrix(kronecker_product(H(), H())) @ Matrix(kronecker_product(Identity(), H())) @ Matrix(kronecker_product(Y(), Z())) @ \
        CX() @ Matrix(kronecker_product(H(), H()))

def solve(quantum_state):
    return SWAP() @ CX() @ SWAP() @ Matrix(kronecker_product(H(), H())) @ Matrix(kronecker_product(Identity(), H())) @ Matrix(kronecker_product(Y(), Z())) @ \
        CX() @ Matrix(kronecker_product(H(), H())) @ quantum_state


initial_state = Matrix([1, 0, 0, 0])
q0 = sympy.Matrix([1, 0])
q1 = sympy.Matrix([1, 0])

print(solve(initial_state))
print(full_system())