import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class CryptoGuard:
    def __init__(self, password, salt):
        self.password = password
        self.salt = salt
        self.kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        self.key = base64.urlsafe_b64encode(self.kdf.derive(self.password.encode()))

    def encrypt(self, data):
        f = Fernet(self.key)
        return f.encrypt(data.encode())

    def decrypt(self, encrypted_data):
        f = Fernet(self.key)
        return f.decrypt(encrypted_data).decode()

# Example usage:
password = "my_secret_password"
salt = os.urandom(16)
crypto_guard = CryptoGuard(password, salt)

api_key = "my_api_key"
encrypted_api_key = crypto_guard.encrypt(api_key)
print("Encrypted API Key:", encrypted_api_key)

decrypted_api_key = crypto_guard.decrypt(encrypted_api_key)
print("Decrypted API Key:", decrypted_api_key)
