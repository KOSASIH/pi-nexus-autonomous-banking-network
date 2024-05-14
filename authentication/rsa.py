from Crypto.Cipher import PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes


class RSACipher:
    def __init__(self, key_size: int = 2048):
        self.key_size = key_size
        self.key = RSA.generate(self.key_size)

    def encrypt(self, data: bytes) -> bytes:
        cipher = PKCS1_OAEP.new(self.key.publickey())
        encrypted_data = cipher.encrypt(data)
        return encrypted_data

    def decrypt(self, encrypted_data: bytes) -> bytes:
        cipher = PKCS1_OAEP.new(self.key)
        decrypted_data = cipher.decrypt(encrypted_data)
        return decrypted_data
