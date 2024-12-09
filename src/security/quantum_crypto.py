from qiskit import QuantumCircuit, execute, Aer

class QuantumCrypto:
    def __init__(self, key_size=256):
        self.key_size = key_size
        self.backend = Aer.get_backend('qasm_simulator')

    def generate_key(self):
        qc = QuantumCircuit(self.key_size)
        for i in range(self.key_size):
            qc.h(i)
        job = execute(qc, self.backend, shots=1)
        result = job.result()
        counts = result.get_counts(qc)
        key = list(counts.keys())[0]
        return key

    def encrypt(self, message, key):
        encrypted_message = ""
        for i in range(len(message)):
            encrypted_message += chr(ord(message[i]) ^ ord(key[i % self.key_size]))
        return encrypted_message

    def decrypt(self, encrypted_message, key):
        decrypted_message = ""
        for i in range(len(encrypted_message)):
            decrypted_message += chr(ord(encrypted_message[i]) ^ ord(key[i % self.key_size]))
        return decrypted_message

# Example usage
if __name__ == "__main__":
    qc = QuantumCrypto(key_size=256)
    key = qc.generate_key()
    message = "Hello, Quantum Crypto!"
    encrypted_message = qc.encrypt(message, key)
    decrypted_message = qc.decrypt(encrypted_message, key)
    print(f"Original Message: {message}")
    print(f"Encrypted Message: {encrypted_message}")
    print(f"Decrypted Message: {decrypted_message}")
