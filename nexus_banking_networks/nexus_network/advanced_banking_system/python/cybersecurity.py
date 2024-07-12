# cybersecurity.py
import hashlib
import hmac

class CyberSecurity:
    def __init__(self):
        self.secret_key = b'secret_key'

    def generate_hash(self, data):
        return hashlib.sha256(data.encode()).hexdigest()

    def verify_hash(self, data, hash):
        return hmac.compare_digest(hash, self.generate_hash(data))

    def encrypt_data(self, data):
        return data.encode() + self.secret_key

    def decrypt_data(self, encrypted_data):
        return encrypted_data[:-len(self.secret_key)].decode()
