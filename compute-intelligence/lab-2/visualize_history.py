import matplotlib.pyplot as plt


def plot_iterations(y_values):
    iterations = list(range(1, len(y_values) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, y_values, marker='o', linestyle='-', color='b')
    plt.title('Values per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Y Value')
    plt.grid(True)
    plt.show()