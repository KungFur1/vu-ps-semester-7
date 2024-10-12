import sympy
from gates_3 import H, Y, Z, T, X


q = sympy.Matrix([1, 0])

result = H() @ Y() @ Z() @ T() @ H() @ X() @ q

print(result)