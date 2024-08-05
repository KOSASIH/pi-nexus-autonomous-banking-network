# quantum_circuit.qasm

OPENQASM 2.0;

// Quantum registers
qreg q[5]; // 5-qubit quantum register
creg c[5]; // 5-bit classical register

// Quantum circuit
gate h q[0]; // Hadamard gate on qubit 0
gate cx q[0], q[1]; // Controlled-NOT gate from qubit 0 to qubit 1
gate ry(π/4) q[2]; // Rotation gate on qubit 2
gate cz q[1], q[3]; // Controlled-Z gate from qubit 1 to qubit 3
gate x q[4]; // Pauli-X gate on qubit 4

// Measurement
measure q -> c; // Measure all qubits and store results in classical register
