import numpy as np
from phe import paillier

class HESSecureDataStorage:
    def __init__(self, public_key, private_key):
        self.public_key = public_key
        self.private_key = private_key

    def encrypt_data(self, data):
        encrypted_data = paillier.encrypt(data, self.public_key)
        return encrypted_data

    def decrypt_data(self, encrypted_data):
        decrypted_data = paillier.decrypt(encrypted_data, self.private_key)
        return decrypted_data

# Example usage:
hes_data_storage = HESSecureDataStorage(paillier.generate_paillier_keypair())
data = 'Hello, Nexus OS!'
encrypted_data = hes_data_storage.encrypt_data(data)
print(f'Encrypted data: {encrypted_data}')

decrypted_data = hes_data_storage.decrypt_data(encrypted_data)
print(f'Decrypted data: {decrypted_data}')
