import numpy as np

class SimpleNeuron:
    def __init__(self, num_inputs):
        # Initialize weights and bias
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand(1)
    
    def sigmoid(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))
    
    def forward(self, inputs):
        # Compute weighted sum of inputs + bias
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        # Apply activation function
        output = self.sigmoid(weighted_sum)
        return output

# Example usage
inputs = np.array([0.5, 0.3, 0.2])  # Example inputs
neuron = SimpleNeuron(num_inputs=len(inputs))

output = neuron.forward(inputs)
print("Neuron output:", output)
