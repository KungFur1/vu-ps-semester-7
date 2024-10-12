import sympy


def vector_probs(v):
    squared_norm = sum([sympy.Abs(component)**2 for component in v])
    return [(sympy.Abs(component)**2) / squared_norm for component in v]


i = sympy.I

v = sympy.Matrix([
    4 + 3*i,
    -5 + 3*i,
    6 - 4*i,
    -4 - 3*i,
    -1 - 5*i,
    0 + 7*i,
    -7 - 5*i,
    0 + 2*i
])

print(vector_probs(v))
print(sum(vector_probs(v)))