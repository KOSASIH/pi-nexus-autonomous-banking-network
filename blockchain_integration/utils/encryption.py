import hashlib
from cryptography.fernet import Fernet

class EncryptionManager:
    def __init__(self, secret_key):
        self.secret_key = secret_key
        self.fernet = Fernet(self.secret_key)

    def encrypt_data(self, data):
        encrypted_data = self.fernet.encrypt(data.encode())
        return encrypted_data

    def decrypt_data(self, encrypted_data):
        decrypted_data = self.fernet.decrypt(encrypted_data).decode()
        return decrypted_data

    def hash_data(self, data):
        hashed_data = hashlib.sha256(data.encode()).hexdigest()
        return hashed_data
