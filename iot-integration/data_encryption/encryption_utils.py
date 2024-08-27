import base64
import hashlib

from cryptography.fernet import Fernet


def generate_secret_key():
    """Generate a secret key for encryption"""
    return Fernet.generate_key()


def encrypt_data(data, secret_key):
    """Encrypt data using a secret key"""
    f = Fernet(secret_key)
    encrypted_data = f.encrypt(data.encode())
    return encrypted_data.decode()


def decrypt_data(encrypted_data, secret_key):
    """Decrypt data using a secret key"""
    f = Fernet(secret_key)
    decrypted_data = f.decrypt(encrypted_data.encode())
    return decrypted_data.decode()


def hash_data(data):
    """Hash data using SHA-256"""
    h = hashlib.sha256()
    h.update(data.encode())
    return h.hexdigest()


def base64_encode(data):
    """Base64 encode data"""
    return base64.b64encode(data.encode()).decode()


def base64_decode(encoded_data):
    """Base64 decode data"""
    return base64.b64decode(encoded_data.encode()).decode()
