import hashlib

def encrypt(data: str, key: str) -> str:
    # Encryption logic here
    encrypted_data = hashlib.sha256((data + key).encode()).hexdigest()
    return encrypted_data

def decrypt(encrypted_data: str, key: str) -> str:
    # Decryption logic here
    decrypted_data = hashlib.sha256((encrypted_data + key).encode()).hexdigest()
    return decrypted_data
