import math


# Single parameter optimization function for previous lab
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
    return I + (L * 0.5)


# # OLD STEEPEST DESCENT FUNC
# def steepest_descent(plot_0, plot_1, step_count=10, alpha=1, rho=0.5, c=1e-2, log=True):
#     history = []
#     for i in range(step_count):
#         history.append((plot_0, plot_1))
#         grad = objective_gradient_function(plot_0, plot_1)
#         step_direction = -grad/numpy.linalg.norm(grad)

#         step_size = alpha
#         while objective_function(plot_0 + step_size * step_direction[0], plot_1 + step_size * step_direction[1]) > objective_function(plot_0, plot_1) - c * step_size * numpy.linalg.norm(grad)**2:
#             step_size *= rho
        
#         plot_0 += step_direction[0] * step_size
#         plot_1 += step_direction[1] * step_size

#         if log:
#             print(f"Plot 0: {plot_0}, Plot 1: {plot_1}, Plot 2: {1 - plot_0 - plot_1}, Objective: {objective_function(plot_0, plot_1)}, Grad: {grad}, Step Size: {step_size}, Step Times Gradient Norm: {step_size * numpy.linalg.norm(grad)}")
        
#     return history