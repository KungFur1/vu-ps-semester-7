import numpy as np
import pandas as pd
import os
import autograd.numpy as np
from autograd import grad
import visualize_history


def sigmoid(A):
    return 1/(1+np.exp(-A))


# X - vertical vector of input values or multpile vertical vectors of input values that form a matrix
# W - vertical vector of input weights
# b - neuron bias float
def neuronFunction(X:np.array, W:np.array, b:float):
    return sigmoid(W.T @ X + b)


def modelFunction(X:np.array, W:np.array, b:float):
    return neuronFunction(X, W, b) # sigmoid(neuronFunction(X, W, b))


def costFunction(X:np.array, Y:np.array, model):
    return np.mean((model(X) - Y) ** 2)


def accuracyFunction(X:np.array, Y:np.array, model):
    return np.sum((np.where(model(X) > 0.5, 1, 0) - Y) == 0) / len(Y)


data_csv = pd.read_csv(os.path.join("refined-data", "iris-combined.csv")) # iris-combined.csv # breast-cancer.csv

X = (data_csv.iloc[:, :-1].values).T
Y = (data_csv.iloc[:,-1].values).T

train_size = int(round(X.shape[1] * 0.8))

X_train = X[:,:train_size]
Y_train = Y[:train_size]

X_test = X[:,train_size:]
Y_test = Y[train_size:]

W = np.random.rand(X.shape[0], 1)
b = np.random.rand()

trainingSetCostFunction = lambda W, b: costFunction(X_train, Y_train, model = lambda X: modelFunction(X, W, b))
trainAccuracyFunction = lambda W, b: accuracyFunction(X_train, Y_train, lambda X: modelFunction(X, W, b))

testSetCostFunction = lambda W, b: costFunction(X_test, Y_test, model = lambda X: modelFunction(X, W, b))
testAccuracyFunction = lambda W, b: accuracyFunction(X_test, Y_test, lambda X: modelFunction(X, W, b))

trainingSetCostGrad_W = grad(trainingSetCostFunction, 0)
trainingSetCostGrad_b = grad(trainingSetCostFunction, 1)

learning_rate = 0.2

train_cost_history = []
test_cost_history = []
test_accuracy_history = []
train_accuracy_history = []
for i in range(10000):
    W_grad = trainingSetCostGrad_W(W, b)
    b_grad = trainingSetCostGrad_b(W, b)
    W -= W_grad * learning_rate
    b -= b_grad * learning_rate

    train_cost = trainingSetCostFunction(W, b)
    test_cost = testSetCostFunction(W, b)
    train_accuracy = trainAccuracyFunction(W, b)
    test_accuracy = testAccuracyFunction(W, b)
    train_cost_history.append(train_cost)
    test_cost_history.append(test_cost)
    test_accuracy_history.append(test_accuracy)
    train_accuracy_history.append(train_accuracy)
    # print(f"Epoch: {i}, train_cost: {train_cost}, test_cost: {test_cost}, test_accuracy: {test_accuracy}")


visualize_history.plot_iterations(train_cost_history)
visualize_history.plot_iterations(train_accuracy_history)
visualize_history.plot_iterations(test_cost_history)
visualize_history.plot_iterations(test_accuracy_history)

print(W)
print(b)
# NEXT: write document, switch datasets (choose which species to predict), make plots, write document...