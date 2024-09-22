import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import binascii

class Secrets:
    def __init__(self):
        self.secret_key = os.environ.get("SECRET_KEY")
        self.jwt_secret_key = os.environ.get("JWT_SECRET_KEY")
        self.database_url = os.environ.get("DATABASE_URL")
        self.fernet_key = self.generate_fernet_key()
        self.kdf = self.generate_kdf()

    def generate_fernet_key(self):
        # Generate a Fernet key
        return Fernet.generate_key()

    def generate_kdf(self):
        # Generate a PBKDF2HMAC key derivation function
        password = os.environ.get("KDF_PASSWORD").encode()
        salt = os.environ.get("KDF_SALT").encode()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return kdf

    def get_secret_key(self):
        return self.secret_key

    def get_jwt_secret_key(self):
        return self.jwt_secret_key

    def get_database_url(self):
        return self.database_url

    def encrypt_data(self, data):
        # Encrypt data using Fernet
        fernet = Fernet(self.fernet_key)
        encrypted_data = fernet.encrypt(data.encode())
        return encrypted_data

    def decrypt_data(self, encrypted_data):
        # Decrypt data using Fernet
        fernet = Fernet(self.fernet_key)
        decrypted_data = fernet.decrypt(encrypted_data).decode()
        return decrypted_data

    def derive_key(self, password):
        # Derive a key using PBKDF2HMAC
        derived_key = self.kdf.derive(password.encode())
        return binascii.hexlify(derived_key).decode()

    def hash_password(self, password):
        # Hash a password using PBKDF2HMAC
        hashed_password = self.kdf.derive(password.encode())
        return binascii.hexlify(hashed_password).decode()

    def verify_password(self, password, hashed_password):
        # Verify a password using PBKDF2HMAC
        derived_key = self.kdf.derive(password.encode())
        return binascii.hexlify(derived_key).decode() == hashed_password

# Example usage:
secrets = Secrets()

# Encrypt data
encrypted_data = secrets.encrypt_data("Hello, World!")
print(encrypted_data)

# Decrypt data
decrypted_data = secrets.decrypt_data(encrypted_data)
print(decrypted_data)

# Derive a key
derived_key = secrets.derive_key("mysecretpassword")
print(derived_key)

# Hash a password
hashed_password = secrets.hash_password("mysecretpassword")
print(hashed_password)

# Verify a password
is_valid = secrets.verify_password("mysecretpassword", hashed_password)
print(is_valid)
