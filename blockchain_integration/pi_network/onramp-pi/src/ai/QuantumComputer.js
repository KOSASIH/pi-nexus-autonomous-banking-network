// QuantumComputer.js

import { Qubit } from 'qubit';

class QuantumComputer {
  constructor() {
    this.qubits = [];
  }

  addQubit(qubit) {
    this.qubits.push(qubit);
  }

  executeGate(gate) {
    // Execute a quantum gate on the qubits
    for (let i = 0; i < this.qubits.length; i++) {
      const qubit = this.qubits[i];
      gate.execute(qubit);
    }
  }

  measure() {
    // Measure the state of the qubits
    const result = [];
    for (let i = 0; i < this.qubits.length; i++) {
      const qubit = this.qubits[i];
      result.push(qubit.measure());
    }
    return result;
  }
}

export default QuantumComputer;
