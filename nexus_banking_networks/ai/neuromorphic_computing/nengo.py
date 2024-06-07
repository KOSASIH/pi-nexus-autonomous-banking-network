import nengo
import numpy as np

class NengoNetwork:
    def __init__(self, num_neurons, num_dimensions):
        self.num_neurons = num_neurons
        self.num_dimensions = num_dimensions
        self.model = nengo.Network()

    def build_network(self):
        with self.model:
            self.input_node = nengo.Node(size_in=self.num_dimensions)
            self.neuron_ensemble = nengo.Ensemble(n_neurons=self.num_neurons, dimensions=self.num_dimensions)
            nengo.Connection(self.input_node, self.neuron_ensemble)

    def run_network(self, input_data):
        with nengo.Simulator(self.model) as sim:
            sim.run(1.0)
            output_data = sim.data[self.neuron_ensemble]
            return output_data

class RealTimeProcessor:
    def __init__(self, nengo_network):
        self.nengo_network = nengo_network

    def process_transaction(self, transaction_data):
        input_data = np.array(transaction_data)
        output_data = self.nengo_network.run_network(input_data)
        return output_data
