import math
import jax
import jax.numpy as jnp


def intervalMethod(objectiveFunction, I:float, r:float, epsilon:float = 10**-4) -> float:
    iteration_count = 0
    # 1
    x_m = (I + r) / 2
    L = r - I
    f_x_m = objectiveFunction(x_m)
    while L >= epsilon:
        # 2
        x_1 = I + L/4
        x_2 = r - L/4
        f_x_1 = objectiveFunction(x_1)
        f_x_2 = objectiveFunction(x_2)
        # 3
        if f_x_1 < f_x_m:
            r = x_m # 3.1
            x_m = x_1 # 3.2
            f_x_m = f_x_1
        # 4
        elif f_x_2 < f_x_m:
            I = x_m # 4.1
            x_m = x_2 # 4.2
            f_x_m = f_x_2
        # 5
        else:
            I = x_1
            r = x_2
        # 6
        L = r - I
        # Extra:
        iteration_count += 1
        print(f"Completed {iteration_count} iterations. I = {I:.5f}, x_1 = {x_1:.5f}, x_m = {x_m:.5f}, x_2 = {x_2:.5f}, r = {r:.5f} | f(x_m) = {objectiveFunction(x_m):.5f}")
    return x_m


def goldenRatioSearchMethod(objectiveFunction, I:float, r:float, epsilon:float = 10**-4) -> float:
    GR = 1/((1 + math.sqrt(5))/2)
    iteration_count:int = 0
    # 1
    L = r - I
    x_1 = r - GR*L
    x_2 = I + GR*L
    f_x_1 = objectiveFunction(x_1)
    f_x_2 = objectiveFunction(x_2)
    # 4
    while L >= epsilon:
        # 2
        if f_x_1 > f_x_2:
            I = x_1
            L = r - I
            x_1 = x_2 # GR optimization
            f_x_1 = f_x_2
            x_2 = I + GR*L
            f_x_2 = objectiveFunction(x_2)
        # 3
        else:
            r = x_2
            L = r - I
            x_2 = x_1 # GR optimization
            f_x_2 = f_x_1
            x_1 = r - GR*L
            f_x_1 = objectiveFunction(x_1)
        # Extra:
        iteration_count += 1
        print(f"Completed {iteration_count} iterations. I = {I:.5f}, x_1 = {x_1:.5f}, x_2 = {x_2:.5f}, r = {r:.5f} | f(x_1) = {objectiveFunction(x_1):.5f} f(x_2) = {objectiveFunction(x_2):.5f}")
    return I + L * 0.5


def newtonsMethod(objectiveFunction, x:float = 5., epsilon:float = 10**-4) -> float:
    first_derivative = jax.grad(objectiveFunction)
    second_derivative = jax.grad(first_derivative)
    iteration_count:int = 0
    step_size = epsilon + 0.1
    while step_size > epsilon:
        step_size = (first_derivative(x)/second_derivative(x))
        x = x - step_size
        iteration_count += 1
        print(f"Completed {iteration_count} iterations. Step size = {step_size:.5f}, x = {x:.5f}, f(x) = {objectiveFunction(x):.5f}, f'(x) = {first_derivative(x):.5f}, f''(x) = {second_derivative(x):.5f}")
    return x