import hashlib
import secrets


class Authentication:
    def __init__(self):
        self.salt = secrets.token_hex(16)

    def generate_password_hash(self, password):
        password_hash = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), self.salt.encode(), 100000
        )
        return password_hash.hex()

    def verify_password(self, password, password_hash):
        password_hash_check = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), self.salt.encode(), 100000
        )
        return password_hash_check.hex() == password_hash
