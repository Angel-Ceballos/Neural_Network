import numpy as np
from base_layer import Layer

class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward(self, output_gradient, learning_rate):
        n = np.size(self.output)
        M = np.tile(self.output, n)
        return np.dot((M * (np.identity(n) - np.transpose(M))), output_gradient)