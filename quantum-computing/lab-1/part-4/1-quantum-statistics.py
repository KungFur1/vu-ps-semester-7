import numpy


# <φ|Aφ>
def quantum_mean(M:numpy.array, v:numpy.array):
    return v.conjugate() @ M @ v


# Var|φi>(A) = 
def quantum_variance(M:numpy.array, v:numpy.array):
    mean = quantum_mean(M, v)
    M_intermediate = M - numpy.identity(M.shape[0]) * mean
    M_intermediate_squared = M_intermediate @ M_intermediate
    return quantum_mean(M_intermediate_squared, v)


A = numpy.array([[1, -1j],
                 [1j, 2]])

phi = numpy.array([1/2, 1j/2]).T

print(quantum_mean(A, phi))

print(quantum_variance(A, phi))