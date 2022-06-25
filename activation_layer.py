from base_layer import Layer
import numpy as np

class Activation(Layer):
    def __init__(self, activation, activation_prime) -> None:
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime
    
    # f(x)
    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    # Calculate the derivative of the error with respect to the input
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))