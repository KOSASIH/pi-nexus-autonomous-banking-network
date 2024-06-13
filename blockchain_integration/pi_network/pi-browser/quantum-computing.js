import { Qiskit } from 'qiskit';

class QuantumComputing {
  constructor() {
    this.qiskit = new Qiskit();
  }

  async runQuantumCircuit(circuit) {
    const result = await this.qiskit.run(circuit);
    return result;
  }
}

export default QuantumComputing;
