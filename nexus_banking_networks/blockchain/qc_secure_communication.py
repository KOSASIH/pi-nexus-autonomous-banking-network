# qc_secure_communication.py
import numpy as np
from quantum_cryptography import QuantumCryptography

class QCSC:
    def __init__(self):
        self.qc = QuantumCryptography()

    def encrypt_data(self, data):
        encrypted_data = self.qc.encrypt(data)
        return encrypted_data

    def decrypt_data(self, encrypted_data):
        decrypted_data = self.qc.decrypt(encrypted_data)
        return decrypted_data

qcsc = QCSC()
