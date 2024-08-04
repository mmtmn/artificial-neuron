import numpy as np

class EnhancedNeuron:
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.weights_exc = np.random.rand(num_inputs)  # Excitatory weights
        self.weights_inh = np.random.rand(num_inputs)  # Inhibitory weights
        self.bias = np.random.rand(1)
        self.last_spike_time = -np.inf
        self.refractory_period = 2.0
        self.learning_rate = 0.01
        self.spike_threshold = 0.5
        self.dendritic_structure = np.random.rand(num_inputs)  # Non-linear summation factors
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def synaptic_plasticity(self, inputs, output):
        # Hebbian learning
        self.weights_exc += self.learning_rate * inputs * output
        self.weights_inh -= self.learning_rate * inputs * output  # Inhibitory synapses learn inversely

    def stdp(self, delta_t):
        # STDP learning rule
        tau = 20.0  # Time constant
        if delta_t > 0:
            return np.exp(-delta_t / tau)
        else:
            return -np.exp(delta_t / tau)
    
    def forward(self, inputs, current_time):
        if current_time - self.last_spike_time < self.refractory_period:
            return 0
        
        # Separate excitatory and inhibitory inputs
        exc_inputs = inputs * (inputs > 0)
        inh_inputs = inputs * (inputs <= 0)
        
        # Non-linear summation for dendritic inputs
        dendritic_exc_sum = np.sum(self.dendritic_structure * exc_inputs ** 2)
        dendritic_inh_sum = np.sum(self.dendritic_structure * inh_inputs ** 2)
        
        # Compute weighted sum
        weighted_sum = np.dot(exc_inputs, self.weights_exc) - np.dot(inh_inputs, self.weights_inh) + dendritic_exc_sum - dendritic_inh_sum + self.bias
        output = self.sigmoid(weighted_sum)
        
        # Check for neuron firing
        if output > self.spike_threshold:
            delta_t = current_time - self.last_spike_time
            self.last_spike_time = current_time
            self.synaptic_plasticity(inputs, output)
            
            # Apply STDP
            for i in range(self.num_inputs):
                self.weights_exc[i] += self.learning_rate * self.stdp(delta_t) * exc_inputs[i]
                self.weights_inh[i] += self.learning_rate * self.stdp(delta_t) * inh_inputs[i]
        
        return output

# Example usage
inputs = np.array([0.5, -0.3, 0.2, -0.1])  # Example inputs with both excitatory and inhibitory signals
neuron = EnhancedNeuron(num_inputs=len(inputs))

# Simulate over time
for t in range(10):
    output = neuron.forward(inputs, current_time=t)
    print(f"Time {t}: Neuron output: {output}")
