import hashlib
import base64

def encrypt_data(data):
    # Encrypt data using a secure encryption algorithm
    encrypted_data = hashlib.sha256(data.encode()).digest()
    return base64.b64encode(encrypted_data)

def decrypt_data(encrypted_data):
    # Decrypt data using a secure decryption algorithm
    decrypted_data = base64.b64decode(encrypted_data)
    return decrypted_data.decode()
