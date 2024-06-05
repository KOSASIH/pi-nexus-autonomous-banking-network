import cryptography

class Encryption:
    def __init__(self, key):
        self.key = key
        self.cipher = cryptography.fernet.Fernet(self.key)

    def encrypt_data(self, data):
        # Encrypt data using Fernet
        encrypted_data = self.cipher.encrypt(data.encode())
        return encrypted_data
