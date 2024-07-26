# dex_project_cybersecurity.py
import hashlib
from cryptography.fernet import Fernet

class DexProjectCybersecurity:
    def __init__(self):
        pass

    def encrypt_data(self, data):
        # Encrypt data using Fernet
        key = Fernet.generate_key()
        cipher_suite = Fernet(key)
        encrypted_data = cipher_suite.encrypt(data.encode())
        return encrypted_data, key

    def decrypt_data(self, encrypted_data, key):
        # Decrypt data using Fernet
        cipher_suite = Fernet(key)
        decrypted_data = cipher_suite.decrypt(encrypted_data)
        return decrypted_data.decode()

    def hash_data(self, data):
        # Hash data using SHA-256
        hash_object = hashlib.sha256()
        hash_object.update(data.encode())
        return hash_object.hexdigest()
