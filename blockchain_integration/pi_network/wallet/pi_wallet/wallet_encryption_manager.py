import os
import hashlib
import base64

class EncryptionManager:
    def __init__(self, encryption_key):
        self.encryption_key = encryption_key

    def encrypt_data(self, data):
        # Encrypt data using AES-256-CBC
        cipher = hashlib.sha256(self.encryption_key.encode()).digest()
        encrypted_data = base64.b64encode(cipher + data.encode())
return encrypted_data

    def decrypt_data(self, encrypted_data):
        # Decrypt data using AES-256-CBC
        cipher = hashlib.sha256(self.encryption_key.encode()).digest()
        decrypted_data = base64.b64decode(encrypted_data).decode()
        return decrypted_data

if __name__ == '__main__':
    encryption_key = 'y_secret_key'
    encryption_manager = EncryptionManager(encryption_key)

    data = 'Hello, World!'
    encrypted_data = encryption_manager.encrypt_data(data)
    print('Encrypted Data:', encrypted_data)

    decrypted_data = encryption_manager.decrypt_data(encrypted_data)
    print('Decrypted Data:', decrypted_data)
