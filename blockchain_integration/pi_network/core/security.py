import hashlib


def encrypt_data(data):
    # Use SHA-256 encryption
    encrypted_data = hashlib.sha256(data.encode()).hexdigest()
    return encrypted_data


def decrypt_data(encrypted_data):
    # Use SHA-256 decryption (not possible, but you can use other encryption methods)
    return encrypted_data
