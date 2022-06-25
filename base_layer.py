class Layer:
    def __init__(self) -> None:
        self.input = None
        self.output = None
    
    def forward(self, input):
        # return output
        pass

    def backward(self, output_gradient, learning_rate):
        # update parameters and return input gradient
        pass
