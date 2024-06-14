import { useState, useEffect } from 'react';
import { useQuantumComputer } from '@qubits/quantum-react';

const QuantumButton = ({ children, onClick }) => {
  const [quantumState, setQuantumState] = useState(0);
  const { executeQuantumGate } = useQuantumComputer();

  useEffect(() => {
    executeQuantumGate('H', quantumState); // Apply Hadamard gate to initialize quantum state
  }, [quantumState]);

  const handleClick = () => {
    executeQuantumGate('X', quantumState); // Apply Pauli-X gate to toggle quantum state
    onClick();
  };

  return (
    <button
      style={{
        backgroundColor: quantumState ? 'blue' : 'red',
        color: quantumState ? 'white' : 'black',
      }}
      onClick={handleClick}
    >
      {children}
    </button>
  );
};

export default QuantumButton;
