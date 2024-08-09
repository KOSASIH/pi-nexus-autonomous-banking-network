/**
 * Quantum Computing Utilities - Constants
 */

export enum QubitType {
  /** Qubit type: Quantum Bit */
  QUBIT,
  /** Qubit type: Classical Bit */
  CBIT,
}

export enum GateType {
  /** Gate type: Hadamard Gate */
  H,
  /** Gate type: Pauli-X Gate */
  X,
  /** Gate type: Pauli-Y Gate */
  Y,
  /** Gate type: Pauli-Z Gate */
  Z,
  /** Gate type: Controlled-NOT Gate */
  CX,
  /** Gate type: Controlled-Phase Shift Gate */
  CZ,
  /** Gate type: Phase Shift Gate */
  S,
  /** Gate type: T Gate */
  T,
  /** Gate type: Measurement Gate */
  MEASURE,
}

export enum QuantumError {
  /** Quantum error: Bit flip error */
  BIT_FLIP,
  /** Quantum error: Phase flip error */
  PHASE_FLIP,
  /** Quantum error: Bit-phase flip error */
  BIT_PHASE_FLIP,
}

export enum QuantumAlgorithm {
  /** Quantum algorithm: Shor's Algorithm */
  SHOR,
  /** Quantum algorithm: Grover's Algorithm */
  GROVER,
  /** Quantum algorithm: Quantum Approximate Optimization Algorithm (QAOA) */
  QAOA,
  /** Quantum algorithm: Variational Quantum Eigensolver (VQE) */
  VQE,
}

export const QUBIT_COUNT = 5; // Default number of qubits
export const GATE_COUNT = 10; // Default number of gates
export const SHOTS = 1024; // Default number of shots for simulation
export const SEED = 42; // Default random seed

export const QASM_SIMULATOR = 'qasm_simulator'; // QASM simulator backend
export const STATEVECTOR_SIMULATOR = 'statevector_simulator'; // Statevector simulator backend
export const QASM_AER = 'qasm_aer'; // QASM Aer backend
export const QISKIT_AER = 'qiskit_aer'; // Qiskit Aer backend

export const QCLOUD_API_KEY = 'YOUR_API_KEY_HERE'; // QCloud API key
export const QCLOUD_API_URL = 'https://api.qcloud.com/v1'; // QCloud API URL
