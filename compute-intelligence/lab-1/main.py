import ui
import ml
import numpy as np
import itertools

# X - objektas, x1,x2,...,xn - pozymiai
# python -m venv myenv
# myenv\Scripts\activate
# pip install numpy
def printResult(W, b, result):
    print("W: ")
    print(W)
    print("b: ")
    print(b)
    print("result: ")
    print(result)



parameter_search_type = ui.getSearchType()
X = np.array([[-0.2, 0.5],[0.2, -0.7],[0.8, 0.8],[0.8, 1]])
Y = np.array([0, 0, 1, 1]).T


activationFunction = ml.heapsviside if (ui.getActivationFunction() == "heapsviside") else ml.sigmoid
W = np.array([0, 0])
b = 0


result_count = 0
W_result_array = []
b_result_array = []
if parameter_search_type == "random":
    while result_count < 5:
        W = np.array([ml.generate_random_float(), ml.generate_random_float()])
        b = ml.generate_random_float()

        result = ml.neuron(X, W, b, activationFunction)
        if (np.array_equal(result, Y)):
            printResult(W,b,result)
            W_result_array.append(W)
            b_result_array.append(b)
            result_count += 1
else:
    print("Iteratively searching for W and b parameters (this might take some time): ")
    for w0, w1, b in itertools.product(np.arange(-20, 20, 0.5), repeat=3):
        W = np.array([w0, w1])
        b = b

        result = ml.neuron(X, W, b, activationFunction)
        if (np.array_equal(result, Y)):
            printResult(W,b,result)
            W_result_array.append(W)
            b_result_array.append(b)
            result_count += 1
            if result_count >= 5:
                break


print("W:")
print(W_result_array)
print("b:")
print(b_result_array)
