# LSP=s2016020; a=2; b=2
import numpy
import math
import visualize
import one_dimension_optimization


# Negative squared volume, assuming total area is equals 1
def objective_function(plot_1, plot_2):
    plot_3 = 1 - plot_1 - plot_2
    if (plot_1 < 0 or plot_2 < 0 or plot_3 < 0):
        return float('inf')
    return -(1/8) * plot_1 * plot_2 * plot_3


def objective_gradient_function(plot_1, plot_2):
    return numpy.array([-1/8*(plot_2 - 2*plot_1*plot_2 - plot_2**2), -1/8*(plot_1 - plot_1**2 - 2*plot_1*plot_2)])


def objective_hesse_matrix_function(plot_1, plot_2):
    h11 = (1/4) * plot_2
    h12 = -1/8 * (1 - 2 * plot_1 - 2 * plot_2)
    h22 = (1/4) * plot_1
    return numpy.array([[h11, h12],
                        [h12, h22]])


def gradient_descent(plot_0, plot_1, step_count=10, learning_rate=1, log=True):
    history = []
    for i in range(step_count):
        history.append((plot_0, plot_1))
        grad = objective_gradient_function(plot_0, plot_1)
        plot_0 -= grad[0] * learning_rate
        plot_1 -=grad[1] * learning_rate

        if log:
            print(f"i = {i} Plot 0: {plot_0}, Plot 1: {plot_1}, Plot 2: {1 - plot_0 - plot_1}, Objective: {objective_function(plot_0, plot_1)}, Grad: {grad}")
            if grad.mean() * learning_rate <= 0.0001:
                print("Small step!")
    
    return history


def steepest_descent(plot_0, plot_1, step_count=3, log=True):
    history = []
    for i in range(step_count):
        history.append((plot_0, plot_1))
        grad = objective_gradient_function(plot_0, plot_1)

        # Find step size
        one_d_objective = lambda step_size: objective_function(plot_0 - grad[0]*step_size, plot_1 - grad[1]*step_size)
        step_size = one_dimension_optimization.goldenRatioSearchMethod(one_d_objective, 0, 100)
        
        plot_0 -= grad[0] * step_size
        plot_1 -= grad[1] * step_size

        if log:
            print(f"i = {i} Plot 0: {plot_0}, Plot 1: {plot_1}, Plot 2: {1 - plot_0 - plot_1}, Objective: {objective_function(plot_0, plot_1)}, Grad: {grad}, Step Size: {step_size}")
            if grad.mean() * step_size:
                print("Small step!")
        
    return history


def simplex_search(point_0=(0.2, 0.2), point_1=(0.21,0.2), point_2=(0.205, 0.21), step_count=10, M=8, scale_factor=2, log=True): # Put the points into an array!
    history = []
    points = [point_0, point_1, point_2]
    point_values = [objective_function(point_0[0], point_0[1]), 
                    objective_function(point_1[0], point_1[1]), 
                    objective_function(point_2[0], point_2[1])]
    reset_counter = [0] * 3

    for i in range(step_count):
        max_index = point_values.index(max(point_values))
        other_indeces = list(range(3))
        other_indeces.remove(max_index)
        highest_p = points[max_index]
        other_p_0 = points[other_indeces[0]]
        other_p_1 = points[other_indeces[1]]
        if any(x > M for x in reset_counter): # Anchoring the highest point (so that I have to write less code..)
            other_p_0 = ((highest_p[0] + other_p_0[0])/2, (highest_p[1] + other_p_0[1])/2)
            other_p_1 = ((highest_p[0] + other_p_1[0])/2, (highest_p[1] + other_p_1[1])/2)

            points[other_indeces[0]] = other_p_0
            points[other_indeces[1]] = other_p_1
            point_values[other_indeces[0]] = objective_function(points[other_indeces[0]][0], points[other_indeces[0]][1])
            point_values[other_indeces[1]] = objective_function(points[other_indeces[1]][0], points[other_indeces[1]][1])

            reset_counter = [0] * 3

            if log:
                print("Shrinked.")
            continue

        center_point = ((other_p_0[0] + other_p_1[0])/2, (other_p_0[1] + other_p_1[1])/2)
        difference = (center_point[0] - highest_p[0], center_point[1] - highest_p[1])

        points[max_index] = (highest_p[0] + difference[0]*2, highest_p[1] + difference[1]*2)
        point_values[max_index] = objective_function(points[max_index][0], points[max_index][1])

        reset_counter[max_index] = 0
        reset_counter[other_indeces[0]] += 1
        reset_counter[other_indeces[1]] += 1

        if point_values[max_index] == min(point_values):
            points[max_index] = (points[max_index][0] + difference[0], points[max_index][1] + difference[1]) # TODO: implement scale factor
            point_values[max_index] = objective_function(points[max_index][0], points[max_index][1])
            if log:
                print("Scaling up.")
        elif point_values[max_index] == max(point_values):
            points[max_index] = (points[max_index][0] - difference[0]*0.5, points[max_index][1] - difference[1]*0.5) # TODO: implement scale factor
            point_values[max_index] = objective_function(points[max_index][0], points[max_index][1])
            if log:
                print("Scaling down.")

        if log:
            print(f"i = {i} Points: {points}, Reset Counter: {reset_counter}")
        
        history.extend(points)
    
    return history


# 6 ---------
# x_0_f = objective_function(0, 0)
# x_0_grad = objective_function(0, 0)

# x_1_f = objective_function(1, 1)
# x_1_grad = objective_function(1, 1)

# x_m_f = objective_function(0.2, 0.2)
# x_m_grad = objective_function(0.2, 0.2)
#   ---------



# history = gradient_descent(0.2, 0.2, 200)
# history = steepest_descent(0.2, 0.2, step_count=1)
# history = simplex_search(step_count=200)

# history = gradient_descent(0, 0) # Stays at 0, 0
# history = gradient_descent(1, 1) # Converges to 1/3

# history = steepest_descent(0, 0) # Stays at nan, nan
# history = steepest_descent(1, 1) # Converges to 1/3


# history = gradient_descent(1, 1, learning_rate=2.666666) # Test
# history = steepest_descent(1.2, 1.2) # Test - Converges from the first iteration

visualize.visualize_2d_function(objective_function, (0,1), (0,1), highlight_points=history)

# With my parameters, I got the best results using steepest descent method. After just 10 iterations the value was already $0.3333337$. Gradient descent after 10 iterations reached a plot 0 value of: $0.281536$. And simplex after 10 iterations: $0.317148$.