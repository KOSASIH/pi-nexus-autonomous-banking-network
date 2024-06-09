import numpy as np

class ScalabilitySolutions:
    def __init__(self):
        self.layer2_scaling = []

    def add_layer2_scaling(self, scaling_solution):
        self.layer2_scaling.append(scaling_solution)

    def optimize_transaction_throughput(self):
        for scaling_solution in self.layer2_scaling:
            scaling_solution.optimize_throughput()

if __name__ == '__main__':
    ss = ScalabilitySolutions()
    layer2_scaling_solution = {'name': 'Optimistic Rollup', 'throughput': 1000}
    ss.add_layer2_scaling(layer2_scaling_solution)

    ss.optimize_transaction_throughput()
