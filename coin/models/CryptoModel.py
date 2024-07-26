import hashlib

class CryptoModel:
    def __init__(self):
        pass

    def generate_key_pair(self) -> (str, str):
        # Generate a new key pair
        private_key = hashlib.sha256("private_key".encode()).hexdigest()
        public_key = hashlib.sha256("public_key".encode()).hexdigest()
        return private_key, public_key

    def encrypt(self, data: str, public_key: str) -> str:
        # Encrypt data using a public key
        encrypted_data = hashlib.sha256(f"{data}{public_key}".encode()).hexdigest()
        return encrypted_data

    def decrypt(self, encrypted_data: str, private_key: str) -> str:
        # Decrypt data using a private key
        decrypted_data = hashlib.sha256(f"{encrypted_data}{private_key}".encode()).hexdigest()
        return decrypted_data
