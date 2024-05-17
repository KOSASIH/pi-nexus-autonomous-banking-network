# pi_nexus/config/config.py
import os

from cryptography.fernet import Fernet


class Config:
    DEBUG = True
    BANKING_API_URL = "https://example.com/banking/data"


config = Config()

CRYPTO_KEY = Fernet.generate_key()
API_ENDPOINT = os.environ.get("API_ENDPOINT", "https://api.example.com")


def encrypt_data(data):
    f = Fernet(CRYPTO_KEY)
    return f.encrypt(data.encode()).decode()


def decrypt_data(data):
    f = Fernet(CRYPTO_KEY)
    return f.decrypt(data.encode()).decode()
