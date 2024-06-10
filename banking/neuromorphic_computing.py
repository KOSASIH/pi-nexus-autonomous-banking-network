import numpy as np
from nengo import Network, Ensemble, Node
from nengo.dists import Uniform
from nengo.utils.ensemble import response_curves

class NeuromorphicComputing:
    def __init__(self):
        self.model = self.create_model()

    def create_model(self):
        model = Network()
        with model:
            ensemble = Ensemble(n_neurons=100, dimensions=1, intercepts=Uniform(-1, 1))
            node = Node(output=lambda t, x: np.sin(x))
            model.connect(ensemble, node)
        return model

    def run_model(self, input_data):
        with self.model:
            self.model.run(input_data)
        return self.model.output

# Example usage:
neuromorphic_computing = NeuromorphicComputing()
input_data = np.random.rand(100, 1)
output = neuromorphic_computing.run_model(input_data)
print(output)
