import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_2d_function(func, x_range=(0, 1), y_range=(0, 1), resolution=100, highlight_points=None):
    # Create a grid of points
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)

    # Initialize an empty array for Z values
    Z = np.zeros_like(X)

    # Compute Z values for each combination of x and y
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(X[i, j], Y[i, j])

    # Handle infinite values by setting them to NaN (optional, helps with visualization)
    Z = np.where(np.isinf(Z), np.nan, Z)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

    # Add color bar for reference
    fig.colorbar(surf)

    # Set labels
    ax.set_xlabel('Plot 1')
    ax.set_ylabel('Plot 2')
    ax.set_zlabel('Objective Function')

    # If highlight points are provided, plot them
    if highlight_points:
        # Separate x and y coordinates from the list of tuples
        highlight_x, highlight_y = zip(*highlight_points)
        
        # Compute the corresponding Z values for the highlight points
        highlight_z = [func(x, y) for x, y in highlight_points]

        # Plot the highlight points in red, with increased size and higher z-order
        ax.scatter(highlight_x, highlight_y, highlight_z, color='red', s=200, marker='o', 
                   edgecolor='black', linewidth=2, zorder=5, label="Highlight Points")

    # Add a legend if highlight points exist
    if highlight_points:
        ax.legend()

    plt.show()

# Example usage:
# highlight_points = [(0.2, 0.3), (0.6, 0.7)]  # Example set of points
# visualize_2d_function(objective_function, x_range=(0, 1), y_range=(0, 1), resolution=100, highlight_points=highlight_points)
