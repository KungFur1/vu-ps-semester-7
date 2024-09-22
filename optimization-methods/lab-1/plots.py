import numpy as np
import matplotlib.pyplot as plt


def visualize(objectiveFunction, interval_x, golden_x, newtons_x):
    interval_x = np.array(interval_x)
    golden_x = np.array(golden_x)
    newtons_x = np.array(newtons_x)
    
    x_plot_min = 0
    x_plot_max = 2

    objective_x = np.linspace(x_plot_min, x_plot_max, 1000)
    objective_y = objectiveFunction(objective_x)
    interval_y = objectiveFunction(interval_x)
    golden_y = objectiveFunction(golden_x)
    newtons_y = objectiveFunction(newtons_x)
    
    plt.figure(figsize=(10, 6))
    plt.ylim([-1.2, 1])
    plt.xlim([x_plot_min, x_plot_max])
    plt.plot(objective_x, objective_y, label="Objective Function")
    plt.plot(interval_x, interval_y, "o", color="red", label="Interval Method")
    plt.plot(golden_x, golden_y, "o", color="yellow", label="Golden Method")
    plt.plot(newtons_x, newtons_y, "o", color="blue", label="Newton's Method")
    
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Objective Function and Optimization History")
    plt.legend()
    plt.grid(True)
    
    plt.show()