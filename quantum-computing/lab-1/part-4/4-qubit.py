import sympy
from gates_3 import H, Y, Z, T, X


q = sympy.Matrix([1, 0])

result = H() @ Y() @ Z() @ T() @ H() @ X() @ q

other = H() @ X() @ X() @ H()


other_other = H() @ H() @ H() @ H() @ H() @ Y() @ X() @ X() @ X() @ Z() @ H() @ H() @ H()

print(other_other) # Turetu grazint vienetine matrica.
print(Y() @ X() @ Z())

#print(result)

# 5HY3XZ3H
# H X X H