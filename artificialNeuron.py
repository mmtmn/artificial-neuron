import numpy as np

class DetailedIonChannel:
    def __init__(self, type, conductance, reversal_potential, state_var):
        self.type = type
        self.conductance = conductance
        self.reversal_potential = reversal_potential
        self.state_var = state_var  # e.g., m, h, n gating variables

    def update_state(self, voltage, time_step):
        # Safely compute alpha and beta
        with np.errstate(over='ignore'):
            alpha = 0.1 * (voltage + 40) / (1 - np.exp(-(voltage + 40) / 10))
            if np.isnan(alpha) or np.isinf(alpha):
                alpha = 0  # Handle invalid alpha values
            
            beta = 4 * np.exp(-(voltage + 65) / 18)
            if np.isnan(beta) or np.isinf(beta):
                beta = 0  # Handle invalid beta values

        self.state_var = self.state_var + time_step * (alpha * (1 - self.state_var) - beta * self.state_var)
        # Ensure state_var stays within valid range
        self.state_var = np.clip(self.state_var, 0, 1)

    def get_current(self, voltage):
        current = self.conductance * (self.state_var ** 3) * (voltage - self.reversal_potential)
        if np.isnan(current) or np.isinf(current):
            current = 0  # Handle invalid current values
        return current

class Synapse:
    def __init__(self, target_neuron, weight, neurotransmitter_type='glutamate'):
        self.target_neuron = target_neuron
        self.weight = weight
        self.neurotransmitter_type = neurotransmitter_type
        self.neurotransmitter_concentration = 0

    def release_neurotransmitter(self):
        self.neurotransmitter_concentration = np.random.rand()  # Probability-based release

    def transmit(self):
        binding = self.neurotransmitter_concentration * np.random.rand()
        self.target_neuron.receive_input(binding * self.weight)

class Astrocyte:
    def __init__(self, support_level):
        self.support_level = support_level

    def modulate(self, inputs):
        return inputs + self.support_level

class DetailedNeuron:
    def __init__(self, num_inputs, params=None):
        if params is None:
            params = {
                'sodium_conductance': 120, 'sodium_reversal_potential': 50, 'sodium_state_var': 0.05,
                'potassium_conductance': 36, 'potassium_reversal_potential': -77, 'potassium_state_var': 0.6,
                'calcium_conductance': 2, 'calcium_reversal_potential': 120, 'calcium_state_var': 0.01,
                'leak_conductance': 0.3, 'leak_reversal_potential': -54.387,
                'membrane_potential': -65, 'threshold_potential': -55, 'reset_potential': -70,
                'membrane_resistance': 10, 'membrane_capacitance': 1, 'time_step': 1,
                'ltp_factor': 0.01, 'ltd_factor': 0.01,
                'refractory_period': 2.0, 'spike_threshold': 0.5, 'learning_rate': 0.01,
                'astrocyte_support_level': 0.1, 'dopamine_level': 1.0
            }
        self.num_inputs = num_inputs
        self.weights_exc = np.random.rand(num_inputs)
        self.weights_inh = np.random.rand(num_inputs)
        self.bias = np.random.rand(1)
        self.last_spike_time = -np.inf
        self.refractory_period = params['refractory_period']
        self.learning_rate = params['learning_rate']
        self.spike_threshold = params['spike_threshold']
        self.dendritic_structure = np.random.rand(num_inputs)
        
        self.sodium_channel = DetailedIonChannel('Na', conductance=params['sodium_conductance'], reversal_potential=params['sodium_reversal_potential'], state_var=params['sodium_state_var'])
        self.potassium_channel = DetailedIonChannel('K', conductance=params['potassium_conductance'], reversal_potential=params['potassium_reversal_potential'], state_var=params['potassium_state_var'])
        self.calcium_channel = DetailedIonChannel('Ca', conductance=params['calcium_conductance'], reversal_potential=params['calcium_reversal_potential'], state_var=params['calcium_state_var'])
        
        self.membrane_potential = params['membrane_potential']
        self.threshold_potential = params['threshold_potential']
        self.reset_potential = params['reset_potential']
        self.membrane_resistance = params['membrane_resistance']
        self.membrane_capacitance = params['membrane_capacitance']
        self.time_step = params['time_step']
        
        self.ltp_factor = params['ltp_factor']
        self.ltd_factor = params['ltd_factor']

        self.astrocyte_support = Astrocyte(support_level=params['astrocyte_support_level'])
        self.dopamine_level = params['dopamine_level']

        self.leak_conductance = params['leak_conductance']
        self.leak_reversal_potential = params['leak_reversal_potential']

        self.axon_length = np.random.rand()
        self.dendritic_tree = np.random.rand(num_inputs)
        self.synapses = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def synaptic_plasticity(self, inputs, output, delta_t):
        if output > self.spike_threshold:
            self.weights_exc += self.ltp_factor * inputs * output
            self.weights_inh -= self.ltp_factor * inputs * output
        else:
            self.weights_exc -= self.ltd_factor * inputs * (1 - output)
            self.weights_inh += self.ltd_factor * inputs * (1 - output)

    def stdp(self, delta_t):
        tau = 20.0
        if delta_t > 0:
            return np.exp(-delta_t / tau)
        else:
            return -np.exp(delta_t / tau)

    def update_ion_channels(self):
        self.sodium_channel.update_state(self.membrane_potential, self.time_step)
        self.potassium_channel.update_state(self.membrane_potential, self.time_step)
        self.calcium_channel.update_state(self.membrane_potential, self.time_step)

    def calculate_total_current(self):
        sodium_current = self.sodium_channel.get_current(self.membrane_potential)
        potassium_current = self.potassium_channel.get_current(self.membrane_potential)
        calcium_current = self.calcium_channel.get_current(self.membrane_potential)
        leak_current = self.leak_conductance * (self.membrane_potential - self.leak_reversal_potential)
        total_current = sodium_current + potassium_current + calcium_current + leak_current
        # Debug statements to trace values
        print(f"Membrane Potential: {self.membrane_potential}")
        print(f"Sodium Current: {sodium_current}")
        print(f"Potassium Current: {potassium_current}")
        print(f"Calcium Current: {calcium_current}")
        print(f"Leak Current: {leak_current}")
        print(f"Total Current: {total_current}")
        return total_current

    def ion_channel_dynamics(self):
        total_current = self.calculate_total_current()
        delta_v = (total_current / self.membrane_resistance) * self.time_step
        if np.isnan(delta_v) or np.isinf(delta_v):
            delta_v = 0  # Handle invalid voltage changes
        self.membrane_potential += delta_v

        if self.membrane_potential >= self.threshold_potential:
            self.membrane_potential = self.reset_potential
            return 1
        else:
            return 0

    def synaptic_transmission(self, neurotransmitter_release):
        receptor_binding = neurotransmitter_release * np.random.rand()
        return receptor_binding

    def intracellular_signaling(self, inputs):
        signaling = np.sum(inputs * self.weights_exc) * np.random.rand()
        return signaling

    def detailed_dendritic_processing(self, inputs):
        processed_inputs = inputs * self.dendritic_tree
        return np.sum(processed_inputs)

    def add_synapse(self, target_neuron, weight, neurotransmitter_type='glutamate'):
        synapse = Synapse(target_neuron, weight, neurotransmitter_type)
        self.synapses.append(synapse)

    def receive_input(self, input_value):
        self.membrane_potential += input_value

    def forward(self, inputs, current_time):
        if current_time - self.last_spike_time < self.refractory_period:
            return 0, 0, 0  # Ensure a tuple is returned

        inputs = self.astrocyte_support.modulate(inputs)
        inputs *= self.dopamine_level

        processed_inputs = self.detailed_dendritic_processing(inputs)

        self.update_ion_channels()
        output = self.ion_channel_dynamics()
        
        if output > self.spike_threshold:
            delta_t = current_time - self.last_spike_time
            self.last_spike_time = current_time
            self.synaptic_plasticity(processed_inputs, output, delta_t)
            
            for i in range(self.num_inputs):
                self.weights_exc[i] += self.learning_rate * self.stdp(delta_t) * processed_inputs
                self.weights_inh[i] += self.learning_rate * self.stdp(delta_t) * processed_inputs
        
        neurotransmitter_release = np.random.rand()
        receptor_binding = self.synaptic_transmission(neurotransmitter_release)
        signaling = self.intracellular_signaling(processed_inputs)
        
        for synapse in self.synapses:
            synapse.release_neurotransmitter()
            synapse.transmit()

        return output, receptor_binding, signaling

class NeuralNetwork:
    def __init__(self, num_neurons, num_inputs, neuron_params=None):
        self.neurons = [DetailedNeuron(num_inputs, params=neuron_params) for _ in range(num_neurons)]

    def connect_neurons(self):
        for neuron in self.neurons:
            target_neuron = np.random.choice(self.neurons)
            weight = np.random.rand()
            neurotransmitter_type = np.random.choice(['glutamate', 'GABA'])
            neuron.add_synapse(target_neuron, weight, neurotransmitter_type)

    def forward(self, inputs, current_time):
        outputs = []
        for neuron in self.neurons:
            output, receptor_binding, signaling = neuron.forward(inputs, current_time)
            outputs.append((output, receptor_binding, signaling))
        return outputs

    def train(self, inputs, targets, epochs):
        for epoch in range(epochs):
            for i in range(len(inputs)):
                current_time = epoch * len(inputs) + i
                outputs = self.forward(inputs[i], current_time)

                for neuron_idx, neuron in enumerate(self.neurons):
                    target = targets[i][neuron_idx]
                    output = outputs[neuron_idx][0]
                    error = target - output

                    # Simple reward-based modulation (dopamine effect)
                    reward = np.exp(-error**2)
                    neuron.dopamine_level = reward

                    for synapse in neuron.synapses:
                        delta_t = current_time - neuron.last_spike_time

                        # Update synaptic weights using STDP and Hebbian learning
                        for j in range(neuron.num_inputs):
                            input_value = float(inputs[i][j])  # Ensure input_value is a scalar
                            stdp_value = neuron.stdp(delta_t)

                            if delta_t > 0:
                                neuron.weights_exc[j] += neuron.learning_rate * stdp_value * input_value * reward
                                neuron.weights_inh[j] -= neuron.learning_rate * stdp_value * input_value * reward
                            else:
                                neuron.weights_exc[j] -= neuron.learning_rate * stdp_value * input_value * reward
                                neuron.weights_inh[j] += neuron.learning_rate * stdp_value * input_value * reward

                            # Apply a Hebbian-like learning rule
                            neuron.weights_exc[j] += neuron.learning_rate * input_value * output
                            neuron.weights_inh[j] -= neuron.learning_rate * input_value * output

                            # Apply bounds to the weights
                            neuron.weights_exc[j] = np.clip(neuron.weights_exc[j], 0, 1)
                            neuron.weights_inh[j] = np.clip(neuron.weights_inh[j], 0, 1)

# Example usage
num_neurons = 10
num_inputs = 5
network = NeuralNetwork(num_neurons, num_inputs)
network.connect_neurons()

inputs = [np.random.rand(num_inputs) for _ in range(10)]  # List of input vectors
targets = [np.random.rand(num_neurons) for _ in range(10)]  # List of target vectors
network.train(inputs, targets, epochs=10)

for t in range(10):
    outputs = network.forward(inputs[t % len(inputs)], current_time=t)
    print(f"Time {t}: Network outputs: {outputs}")
