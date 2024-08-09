from qiskit_simulator import QiskitSimulator

def get_qc_for_n_qubit_GHZ_state(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc

def simulate_GHZ_state(n: int):
    simulator = QiskitSimulator(n)
    qc = get_qc_for_n_qubit_GHZ_state(n)
    simulator.circuit = qc
    result = simulator.run()
    return result
