import notes
import autograd.numpy as np
from autograd import grad


# Negative squared volume
def f(plot_1, plot_2, plot_3):
    return -(1/8) * plot_1 * plot_2 * plot_3


def g_1(plot_1, plot_2, plot_3):
    return plot_1 + plot_2 + plot_3 - 1


def h_1(plot_1, plot_2, plot_3): # -plot_1 <= 0
    return -plot_1


def h_2(plot_1, plot_2, plot_3): # -plot_2 <= 0
    return -plot_2


def h_3(plot_1, plot_2, plot_3): # -plot_3 <= 0
    return -plot_3


def f_with_restrictions(plot_1, plot_2, plot_3):
    r = 1
    return f(plot_1, plot_2, plot_3) + (1/r) * (g_1(plot_1, plot_2, plot_3)**2 + 
                                                max(0, h_1(plot_1, plot_2, plot_3))**2 + 
                                                max(0, h_2(plot_1, plot_2, plot_3))**2 + 
                                                max(0, h_3(plot_1, plot_2, plot_3))**2)


def gradient_f_with_restrictions(plot_1, plot_2, plot_3, r=1):
    x = plot_1
    y = plot_2
    z = plot_3

    df_dx = - (1/8) * y * z
    df_dy = - (1/8) * x * z
    df_dz = - (1/8) * x * y

    g_1 = x + y + z - 1
    dg_1_dx = 1
    dg_1_dy = 1
    dg_1_dz = 1

    dP_dx = 2 * x if x < 0 else 0
    dP_dy = 2 * y if y < 0 else 0
    dP_dz = 2 * z if z < 0 else 0

    grad_penalty_x = 2 * g_1 * dg_1_dx + dP_dx
    grad_penalty_y = 2 * g_1 * dg_1_dy + dP_dy
    grad_penalty_z = 2 * g_1 * dg_1_dz + dP_dz

    grad_x = df_dx + (1 / r) * grad_penalty_x
    grad_y = df_dy + (1 / r) * grad_penalty_y
    grad_z = df_dz + (1 / r) * grad_penalty_z

    return grad_x, grad_y, grad_z


def gradient_descent(grad_func, objective_func, X, step_count=1500, learning_rate=0.1, log=True):
    plot_1, plot_2, plot_3 = X
    history = []
    r = 1
    for y in range(5):
        for i in range(step_count):
            history.append(X)
            grad = grad_func(plot_1, plot_2, plot_3, r)
            plot_1 -= grad[0] * learning_rate
            plot_2 -= grad[1] * learning_rate
            plot_3 -= grad[2] * learning_rate
        if log:
            print(f"iteration = {i * step_count} X: {plot_1}, {plot_2}, {plot_3}. Objective: {objective_func(plot_1, plot_2, plot_3)}, bauda = {(g_1(plot_1, plot_2, plot_3)**2 + 
                                                max(0, h_1(plot_1, plot_2, plot_3))**2 + 
                                                max(0, h_2(plot_1, plot_2, plot_3))**2 + 
                                                max(0, h_3(plot_1, plot_2, plot_3))**2)}")
        r /= 2
        learning_rate /= 2 #(1/r) * 1/r kart 

        print(f"r = {r}")

    return history


X_0 = (0, 0, 0)
X_1 = (1, 1, 1)
X_m = (0/10, 2/10, 0/10)

gradient_descent(gradient_f_with_restrictions, f_with_restrictions, X_0)


# print(f"f(0, 0, 0)={f(0, 0, 0)}")
# print(f"f(1, 1, 1)={f(1, 1, 1)}")
# print(f"f(0, 2/10, 0)={f(0/10, 2/10, 0/10)}")

# print(f"g_1(0, 0, 0)={g_1(0, 0, 0)}")
# print(f"g_1(1, 1, 1)={g_1(1, 1, 1)}")
# print(f"g_1(0, 2/10, 0)={g_1(0/10, 2/10, 0/10)}")

# print(f"h_1,h_2,h_3(0, 0, 0)={h_1(0, 0, 0)}, {h_2(0, 0, 0)}, {h_3(0, 0, 0)}")
# print(f"h_1,h_2,h_3(1, 1, 1)={h_1(1, 1, 1)}, {h_2(1, 1, 1)}, {h_3(1, 1, 1)}")
# print(f"h_1,h_2,h_3(0, 2/10, 0)={h_1(0, 2/10, 0)}, {h_2(0, 2/10, 0)}, {h_3(0, 2/10, 0)}")