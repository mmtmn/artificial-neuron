import numpy as np

class AdvancedNeuron:
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand(1)
        self.last_spike_time = -np.inf  # Initialize last spike time to negative infinity
        self.refractory_period = 2.0  # Refractory period in arbitrary time units
        self.learning_rate = 0.01  # Learning rate for synaptic plasticity
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def synaptic_plasticity(self, inputs, output):
        # Adjust weights based on activity (Hebbian learning)
        self.weights += self.learning_rate * inputs * output
    
    def forward(self, inputs, current_time):
        # Check for refractory period
        if current_time - self.last_spike_time < self.refractory_period:
            return 0
        
        # Compute weighted sum of inputs + bias
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        # Apply activation function
        output = self.sigmoid(weighted_sum)
        
        # If neuron fires (output > threshold), update last spike time
        if output > 0.5:  # Assuming 0.5 as a firing threshold
            self.last_spike_time = current_time
            # Apply synaptic plasticity
            self.synaptic_plasticity(inputs, output)
        
        return output

# Example usage
inputs = np.array([0.5, 0.3, 0.2])  # Example inputs
neuron = AdvancedNeuron(num_inputs=len(inputs))

# Simulate over time
for t in range(10):
    output = neuron.forward(inputs, current_time=t)
    print(f"Time {t}: Neuron output: {output}")
