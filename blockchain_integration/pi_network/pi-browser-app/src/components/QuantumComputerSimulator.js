import React, { useState, useEffect } from 'react';
import { QuantumComputerAPI } from '../api';
import { Qubit, QuantumGate } from 'quantum-computing-js';

const QuantumComputerSimulator = () => {
  const [qubits, setQubits] = useState([new Qubit(0), new Qubit(0)]);
  const [gates, setGates] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchQubits = async () => {
      const response = await QuantumComputerAPI.getQubits();
      setQubits(response.data);
      setLoading(false);
    };
    fetchQubits();
  }, []);

  const handleAddGate = (gate) => {
    setGates([...gates, gate]);
  };

  const handleRunSimulation = async () => {
    setLoading(true);
    const response = await QuantumComputerAPI.runSimulation(qubits, gates);
    setResult(response.data);
    setLoading(false);
  };

  return (
    <div className="quantum-computer-simulator">
      <h1>Quantum Computer Simulator</h1>
      <p>Qubits:</p>
      <ul>
        {qubits.map((qubit, index) => (
          <li key={index}>
            <p>Qubit {index}: {qubit.state}</p>
          </li>
        ))}
      </ul>
      <p>Gates:</p>
      <ul>
        {gates.map((gate, index) => (
          <li key={index}>
            <p>Gate {index}: {gate.type}</p>
          </li>
        ))}
      </ul>
      <button onClick={handleRunSimulation}>Run Simulation</button>
      {loading ? (
        <p>Loading...</p>
      ) : (
        <p>Result: {result}</p>
      )}
      <QuantumGateSelector onAddGate={handleAddGate} />
    </div>
  );
};

const QuantumGateSelector = ({ onAddGate }) => {
  const [gateType, setGateType] = useState('H');

  const handleGateTypeChange = (event) => {
    setGateType(event.target.value);
  };

  const handleAddGateClick = () => {
    const gate = new QuantumGate(gateType);
    onAddGate(gate);
  };

  return (
    <div className="quantum-gate-selector">
      <label>
        Gate Type:
        <select value={gateType} onChange={handleGateTypeChange}>
          <option value="H">Hadamard Gate</option>
          <option value="X">Pauli-X Gate</option>
          <option value="Y">Pauli-Y Gate</option>
          <option value="Z">Pauli-Z Gate</option>
        </select>
      </label>
      <button onClick={handleAddGateClick}>Add Gate</button>
    </div>
  );
};

export default QuantumComputerSimulator;
