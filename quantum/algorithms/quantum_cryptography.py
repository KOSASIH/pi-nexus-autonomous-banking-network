# quantum_cryptography.py
import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, assemble, execute
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector

class QuantumDigitalSignature:
    def __init__(self, message):
        """
        Initialize the Quantum Digital Signature protocol.

        Parameters:
        - message: The message to be signed.
        """
        self.message = message
        self.key = None

    def generate_key(self):
        """Generate a random key for signing."""
        self.key = np.random.randint(0, 2, len(self.message))

    def sign(self):
        """
        Sign the message using the generated key.

        Returns:
        - signed_message: The signed message as a quantum state.
        """
        if self.key is None:
            raise ValueError("Key not generated. Call generate_key() first.")

        # Create a quantum circuit for signing
        circuit = QuantumCircuit(len(self.message))

        # Encode the message and key into the quantum state
        for i in range(len(self.message)):
            if self.message[i] == '1':
                circuit.x(i)  # Apply X gate for '1' bits

            if self.key[i] == 1:
                circuit.h(i)  # Apply Hadamard gate for key bits

        # Measure the signed message
        circuit.measure_all()

        # Execute the circuit
        backend = Aer.get_backend('qasm_simulator')
        transpiled_circuit = transpile(circuit, backend)
        qobj = assemble(transpiled_circuit)
        result = execute(qobj, backend, shots=1024).result()
        counts = result.get_counts()

        return counts

class QuantumSecretSharing:
    def __init__(self, secret, n_shares):
        """
        Initialize the Quantum Secret Sharing protocol.

        Parameters:
        - secret: The secret to be shared (binary string).
        - n_shares: Number of shares to create.
        """
        self.secret = secret
        self.n_shares = n_shares

    def share_secret(self):
        """
        Share the secret among n participants.

        Returns:
        - shares: List of shares for each participant.
        """
        # Create a quantum circuit for sharing the secret
        circuit = QuantumCircuit(len(self.secret) + self.n_shares)

        # Encode the secret into the quantum state
        for i in range(len(self.secret)):
            if self.secret[i] == '1':
                circuit.x(i)  # Apply X gate for '1' bits

        # Create shares using entanglement
        for i in range(len(self.secret)):
            for j in range(self.n_shares):
                circuit.cx(i, len(self.secret) + j)  # CNOT to create shares

        # Measure the shares
        circuit.measure_all()

        # Execute the circuit
        backend = Aer.get_backend('qasm_simulator')
        transpiled_circuit = transpile(circuit, backend)
        qobj = assemble(transpiled_circuit)
        result = execute(qobj, backend, shots=1024).result()
        counts = result.get_counts()

        # Extract shares from the measurement results
        shares = []
        for i in range(self.n_shares):
            share = ''.join([counts.get(f"{i:0{len(self.secret)}b}", 0) for i in range(2**len(self.secret))])
            shares.append(share)

        return shares

if __name__ == "__main__":
    # Example usage for Quantum Digital Signature
    message = "1011"
    qds = QuantumDigitalSignature(message)
    qds.generate_key()
    signed_message = qds.sign()
    print("Signed Message Counts:", signed_message)

    # Example usage for Quantum Secret Sharing
    secret = "1101"
    n_shares = 3
    qss = QuantumSecretSharing(secret, n_shares)
    shares = qss.share_secret()
    print("Shares:", shares)
