import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from dense_layer import Dense
from convolution_layer import Convolutional
from reshape_layer import Reshape
from activations import Tanh, Sigmoid
from softmax_layer import Softmax
from losses import binary_cross_entropy, binary_cross_entropy_prime

def preprocess_data(x, y, limit):
    number_list = np.zeros(shape=(10, limit), dtype=np.int32)
    for i in range(10):
        number_list[i][:] = np.where(y == i)[0][:limit]

    all_indices = np.hstack(number_list)
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 10, 1)
    return x, y

# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

# neural network
network = [
    Convolutional((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 10),
    Softmax()
]

epochs = 30
learning_rate = 0.1

# Training
for e in range(epochs):
    error = 0
    for x, y in zip(x_train, y_train):
        # forward propagation
        output = x
        for layer in network:
            output = layer.forward(output)

        # error on prediction
        error += binary_cross_entropy(y, output)

        # backward propagation
        grad = binary_cross_entropy_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)

        
    error /= len(x_train)
    print('%d/%d, error=%f' % (e + 1, epochs, error))

# Pedrict
for x, y in zip(x_test, y_test):
    # forward propagation
    output = x
    for layer in network:
        output = layer.forward(output)
    prediction = output
    print(f'True:{np.argmax(y)}, Prediction:{np.argmax(prediction)}')