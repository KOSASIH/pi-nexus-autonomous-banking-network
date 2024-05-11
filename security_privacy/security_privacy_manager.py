import random
import numpy as np
from security_privacy.homomorphic_encryption import HomomorphicEncryption

class SecurityPrivacyManager:
    def __init__(self, homomorphic_encryption):
        self.homomorphic_encryption = homomorphic_encryption

    def encrypt_data(self, data):
        """
        Encrypts user data using homomorphic encryption.
        """
        public_key, _ = self.homomorphic_encryption.generate_keys()

        encrypted_data = []
        for value in data:
            encrypted_value = self.homomorphic_encryption.encrypt(public_key, value)
            encrypted_data.append(encrypted_value)

        return encrypted_data

    def perform_computations(self, encrypted_data, operations):
        """
        Performs computations on encrypted data using homomorphic encryption.
        """
        public_key = self.homomorphic_encryption.public_key

        for operation in operations:
            if operation['operation'] == 'add':
                ciphertext1, ciphertext2 = encrypted_data[operation['index1']], encrypted_data[operation['index2']]
                encrypted_data[operation['result_index']] = self.homomorphic_encryption.add(public_key, ciphertext1, ciphertext2)
            elif operation['operation'] == 'multiply':
                ciphertext = encrypted_data[operation['index']]
                value = operation['value']
                encrypted_data[operation['result_index']] = self.homomorphic_encryption.multiply(public_key, ciphertext, value)

        return encrypted_data

    def decrypt_results(self, encrypted_data, private_key):
        """
        Decrypts the results of the computations using the private key.
        """
        decrypted_data = []
        for ciphertext in encrypted_data:
            decrypted_value = self.homomorphic_encryption.decrypt(private_key, ciphertext)
            decrypted_data.append(decrypted_value)

        return decrypted_data
