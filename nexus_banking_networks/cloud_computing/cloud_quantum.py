import qiskit
from qiskit import QuantumCircuit, execute

def create_quantum_circuit(qubit_count):
    # Create a new quantum circuit
    circuit = QuantumCircuit(qubit_count)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    return circuit

def execute_quantum_circuit(circuit, backend):
    # Execute the quantum circuit on a cloud backend
    job = execute(circuit, backend, shots=1024)
    result = job.result()
    return result.get_counts(circuit)

if __name__ == '__main__':
    qubit_count = 2
    backend = 'ibmq_qasm_simulator'

    circuit = create_quantum_circuit(qubit_count)
    result = execute_quantum_circuit(circuit, backend)
    print(f"Quantum circuit executed with result: {result}")
