import pyotp
from cryptography.fernet import Fernet

class SMFA:
    def __init__(self, secret_key_path):
        self.secret_key = open(secret_key_path, 'rb').read()
        self.fernet = Fernet(self.secret_key)

    def generate_otp(self, user_id):
        totp = pyotp.TOTP(self.secret_key)
        otp = totp.now()
        return otp

    def verify_otp(self, user_id, otp):
        totp = pyotp.TOTP(self.secret_key)
        return totp.verify(otp)

    def encrypt_data(self, data):
        encrypted_data = self.fernet.encrypt(data.encode())
        return encrypted_data

    def decrypt_data(self, encrypted_data):
        decrypted_data = self.fernet.decrypt(encrypted_data).decode()
        return decrypted_data
