from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import numpy as np

def quantum_encryption(message):
    # Convert the message to binary
    binary_message = ''.join(format(ord(char), '08b') for char in message)
    message_length = len(binary_message)
    
    # Step 1: Generate a random key of the same length as the message
    key = np.random.randint(2, size=message_length)
    
    # Step 2: Create a quantum circuit for encryption
    qc = QuantumCircuit(message_length, message_length)
    
    # Step 3: Prepare the message in the quantum circuit
    for i, bit in enumerate(binary_message):
        if bit == '1':
            qc.x(i)  # Prepare |1> state
    
    # Step 4: Apply the key using XOR operation
    for i in range(message_length):
        if key[i] == 1:
            qc.x(i)  # Apply X gate if key bit is 1
    
    # Step 5: Measure the encrypted message
    qc.measure(range(message_length), range(message_length))
    
    # Step 6: Execute the circuit
    backend = Aer.get_backend('qasm_simulator')
    result = execute(qc, backend, shots=1024).result()
    counts = result.get_counts()
    
    # Step 7: Plot the results
    plot_histogram(counts).show()
    
    # Step 8: Return the encrypted message and the key
    encrypted_message = list(counts.keys())[0]  # Get the most frequent measurement result
    return encrypted_message, key

if __name__ == "__main__":
    message = "Hello"
    encrypted_message, key = quantum_encryption(message)
    print("Original Message:", message)
    print("Encrypted Message:", encrypted_message)
    print("Key Used:", key)
