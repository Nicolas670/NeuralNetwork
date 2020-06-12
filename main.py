import numpy as np
import nnfs
from nnfs.datasets import spiral_data

np.random.seed(0)

X = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(100,3)


class LayerDense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

layer1 = LayerDense(2, 5)
activation1 = ActivationReLU()

layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output)

""""
inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26,  -0.5], [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

# Manual dotproduct
def calculate_output(inputs, neuron_weights, neuron_biases):
    outputs = []
    for neuron_weights, bias in zip(weights, biases):
        output = 0
        for weight, input in zip(neuron_weights, inputs):
            output += weight*input
        output += bias

        outputs.append(output)

    return outputs

# With Numpy
output = np.dot(weights, inputs) + biases
#output = calculate_output(inputs, weights, biases)
print(output)
"""