from re import L
from base_layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        # weight matrix (j * i)
        self.weights = np.random.randn(output_size, input_size)
        # bias matrix (j * 1)
        self.bias = np.random.randn(output_size, 1)

    # Y = W*X+ B
    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        # calculate the derivative of the error with respect to the weights
        weights_gradient = np.dot(output_gradient, self.input.T)
        # calculate and return the derivative of the error with respect to the input
        input_gradient = np.dot(self.weights.T, output_gradient)
        # update weights and bias with gradient decent
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient