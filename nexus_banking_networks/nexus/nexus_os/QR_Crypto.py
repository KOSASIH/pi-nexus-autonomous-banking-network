import numpy as np
from qiskit import QuantumCircuit, execute

class QRCrypto:
    def __init__(self, key_size):
        self.key_size = key_size
        self.qc = QuantumCircuit(key_size)

    def generate_key(self):
        self.qc.h(range(self.key_size))
        self.qc.measure_all()
        result = execute(self.qc, backend='qasm_simulator').result()
        key = ''.join([str(bit) for bit in result.get_counts()])
        return key

    def encrypt(self, plaintext, key):
        ciphertext = ''
        for i, char in enumerate(plaintext):
            ciphertext += chr(ord(char) ^ ord(key[i % len(key)]))
        return ciphertext

    def decrypt(self, ciphertext, key):
        plaintext = ''
        for i, char in enumerate(ciphertext):
            plaintext += chr(ord(char) ^ ord(key[i % len(key)]))
        return plaintext

# Example usage:
qr_crypto = QRCrypto(256)
key = qr_crypto.generate_key()
print(f'Generated key: {key}')

plaintext = 'Hello, Nexus OS!'
ciphertext = qr_crypto.encrypt(plaintext, key)
print(f'Encrypted: {ciphertext}')

decrypted = qr_crypto.decrypt(ciphertext, key)
print(f'Decrypted: {decrypted}')
