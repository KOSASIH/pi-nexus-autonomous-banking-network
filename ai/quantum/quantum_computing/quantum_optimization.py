import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.optimization import QuadraticProgram
from qiskit.optimization.algorithms import VQE, QAOA

class QuantumOptimization:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.circuit = QuantumCircuit(num_qubits)

    def define_quadratic_program(self, matrix, offset):
        qp = QuadraticProgram()
        qp.from_ising(matrix, offset, linear=True)
        return qp

    def run_vqe(self, qp, backend='qasm_simulator'):
        vqe = VQE(qp, self.circuit, backend)
        result = vqe.run()
        return result

    def run_qaoa(self, qp, backend='qasm_simulator'):
        qaoa = QAOA(qp, self.circuit, backend)
        result = qaoa.run()
        return result

    def plot_optimization(self, result):
        import matplotlib.pyplot as plt
        plt.plot(result.eigenvalues)
        plt.xlabel('Iteration')
        plt.ylabel('Energy')
        plt.title('Quantum Optimization Results')
        plt.show()

# Example usage
optimization = QuantumOptimization(4)
matrix = np.array([[1, 2, 3, 4], [2, 5, 6, 7], [3, 6, 8, 9], [4, 7, 9, 10]])
offset = 10
qp = optimization.define_quadratic_program(matrix, offset)
result = optimization.run_vqe(qp)
optimization.plot_optimization(result)
