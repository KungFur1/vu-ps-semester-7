import numpy


def softmax(x):
    x_shifted = x - numpy.max(x, axis=0, keepdims=True)
    e_x = numpy.exp(x_shifted)
    return e_x / e_x.sum(axis=0, keepdims=True)


def sigmoid(X):
    return 1 / (1 + numpy.exp(-X))


# Each column represents a data point (each row a feature)
X = numpy.genfromtxt('iris-new-test.csv', delimiter=",")[:, :-1].T
min_vals = X.min(axis=0)
max_vals = X.max(axis=0)
X = (2 * X - min_vals - max_vals) / (max_vals - min_vals)

# Each row represents a neuron (each column a specific features weights)
W0 = numpy.array([[1.0094239000969418, -2.8861984211733054, 3.524109890797802],
                 [-0.6033113487383684, 0.25333881908187433, -3.6220367976542143],
                 [-1.8152549194443377, 3.1348671687767653, -5.417135204096832],
                 [0.25354873129701394, 1.690814522158558, -8.296843560887408],
                 [1.600551815234887, -2.841080599997625, 6.556426248370086]])
b0 = numpy.array([[1.3195263166421949], 
                  [1.1195114721466286], 
                  [-2.8805678200060667], 
                  [3.9195462572571373], 
                  [-2.4707970148546785]])

W1 = numpy.array([[-5.155589087569882, 0.25303286914960293, 4.09097683452935, 1.5018299785124933, -3.201927916423887],
                 [1.8726064573393189, 1.0937839520407748, -7.831382567951255, 5.169787408661677, -5.3936820851078355],
                 [3.7599418891368646, -2.341082920843232, -3.7328746654661598, -5.198485427952334, 3.970058776658249]])
b1 = numpy.array([[-1.0107020526599064], 
                  [-2.942710079399252], 
                  [-0.9416790197108846]])


print(X.shape)
print(W0.shape)
Y = softmax(W1 @ sigmoid(W0 @ X + b0) +b1)
numpy.set_printoptions(suppress=True, precision=8)
print(Y.T)
Y = numpy.argmax(Y, axis=0).T
print(Y)
class_names = numpy.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
Y = class_names[Y]
print(Y)