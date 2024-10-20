import numpy
from sympy import Matrix, I, exp, pi, sqrt
from sympy import symbols

# Pakeisti pavyzdiui posukio matricoj gali keistis kampas priklausomai nuo laiko
def apply_time(time):
    return Matrix([
        [1, 0],
        [0, exp(I * time)]
    ])


print(apply_time(5))