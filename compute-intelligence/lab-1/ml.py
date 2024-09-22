import numpy as np
import random

def neuron(X, W, b, activationFunction):
    return activationFunction(X @ W.T + b)

def heapsviside(A):
    return np.where(A >= 0, 1, 0)

def sigmoid(A):
    return np.where(1/(1+np.exp(-A)) >= 0.5, 1, 0)

def generate_random_float():
    return round(random.uniform(-20, 20), 1)