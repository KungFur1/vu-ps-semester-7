import sympy as sp


def normalize_vector(vector):
    if not (isinstance(vector, sp.Matrix) and (vector.shape[0] == 1 or vector.shape[1] == 1)):
        raise ValueError("Input must be a SymPy Matrix representing a vector (either column or row).")
    
    # Calculate the squared norm (sum of |component|^2)
    squared_norm = sum([sp.Abs(component)**2 for component in vector])
    
    # Handle zero vector
    if squared_norm == 0:
        raise ZeroDivisionError("Cannot normalize a zero vector.")
    
    # Calculate the norm
    norm = sp.sqrt(squared_norm)
    
    # Normalize the vector by dividing each component by the norm
    normalized_vector = vector / norm
    
    return normalized_vector
i = sp.I

v1 = sp.Matrix([
    4 + 3*i,
    -5 + 3*i,
    6 - 4*i,
    -4 - 3*i,
    -1 - 5*i,
    0 + 7*i,
    -7 - 5*i,
    0 + 2*i
])


v2 = sp.Matrix([
    -2 - 5*i,
    0 + 3*i,     # 3i is represented as 0 + 3i
    7 - 1*i,
    -2 + 4*i,
    0 + 4*i,     # 4i is represented as 0 + 4i
    -1 - 3*i,
    -3 - 5*i,
    1 - 5*i
])

print(v1)
print(v2)

print(normalize_vector(v1))
print(normalize_vector(v2))

v1 = normalize_vector(v1)
v2 = normalize_vector(v2)

answer = v1.dot(v2)/(v1.norm() * v2.norm())

print(answer.simplify())