import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


M = np.array([
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
])


def apply_state_change(vector, k, M):
    for step in range(k):
        vector = M @ vector
    return vector


def vsualize_system(M):
    G = nx.DiGraph()

    G.add_nodes_from(list(range(M.shape[0])))

    for row in range(M.shape[0]):
        for col in range(M.shape[1]):
            if M[row][col] == 1:
                G.add_edge(col, row)


    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    plt.title("System Dynamics Visualization")
    plt.axis('off')
    plt.show()


k = 9
initial_state = np.zeros(10)
initial_state[4] = 1
final_state = apply_state_change(initial_state, k, M)
print(f"Initial State:\n{initial_state}")
print(f"Final State after {k} steps:\n{final_state}")

vsualize_system(M)