import numpy


def apply_time(M, t):
    for i in range(t):
        M = M @ M
    return M


M = numpy.array([
    [ 0.2083+0.0000j,  0.3391-0.0991j,  0.4182+0.1095j, -0.1932+0.1585j],
    [ 0.5584+0.0000j, -0.1035+0.2334j, -0.1115-0.0563j,  0.1512-0.1967j],
    [ 0.3902+0.0000j,  0.0413+0.2104j, -0.3265+0.1592j,  0.0953+0.0895j],
    [ 0.6749+0.0000j, -0.0502-0.0631j,  0.1273+0.1204j, -0.3848-0.1112j]
], dtype=complex)

v = numpy.array([[1, 0, 0, 0]], dtype=complex).T

print(apply_time(M, 4) @ v)