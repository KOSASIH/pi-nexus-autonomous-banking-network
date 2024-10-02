import base64
import os

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


def generate_key(password, salt):
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key


def encrypt_data(key, data):
    cipher_suite = Fernet(key)
    encrypted_data = cipher_suite.encrypt(data.encode())
    return encrypted_data


def decrypt_data(key, encrypted_data):
    cipher_suite = Fernet(key)
    decrypted_data = cipher_suite.decrypt(encrypted_data).decode()
    return decrypted_data


def generate_salt():
    salt = os.urandom(16)
    return salt


def main():
    password = "my_secret_password"
    salt = generate_salt()
    key = generate_key(password, salt)
    data = "This is some secret data"
    encrypted_data = encrypt_data(key, data)
    print("Encrypted data:", encrypted_data)
    decrypted_data = decrypt_data(key, encrypted_data)
    print("Decrypted data:", decrypted_data)


if __name__ == "__main__":
    main()
