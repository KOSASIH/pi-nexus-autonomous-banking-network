# dex_project_cybersecurity.py
import hashlib
import os

class DexProjectCybersecurity:
    def __init__(self):
        pass

    def encrypt_data(self, data, password):
        # Encrypt data using AES
        from cryptography.fernet import Fernet
        key = hashlib.sha256(password.encode()).digest()
        cipher_suite = Fernet(key)
        encrypted_data = cipher_suite.encrypt(data.encode())
        return encrypted_data

    def decrypt_data(self, encrypted_data, password):
        # Decrypt data using AES
        from cryptography.fernet import Fernet
        key = hashlib.sha256(password.encode()).digest()
        cipher_suite = Fernet(key)
        decrypted_data = cipher_suite.decrypt(encrypted_data).decode()
        return decrypted_data

    def generate_random_password(self, length):
        # Generate a random password
        import secrets
        alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'
        password = ''.join(secrets.choice(alphabet) for _ in range(length))
        return password

    def check_password_strength(self, password):
        # Check the strength of a password
        import re
        if len(password) < 8:
            return False
        if not re.search('[a-z]', password):
            return False
        if not re.search('[A-Z]', password):
            return False
        if not re.search('[0-9]', password):
            return False
        return True
