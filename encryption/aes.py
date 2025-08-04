from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes


class AESCipher:
    def __init__(self, key: bytes):
        self.key = key

    def encrypt(self, data: bytes) -> bytes:
        iv = get_random_bytes(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CFB, iv)
        encrypted_data = iv + cipher.encrypt(data)
        return encrypted_data

    def decrypt(self, encrypted_data: bytes) -> bytes:
        iv = encrypted_data[: AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CFB, iv)
        decrypted_data = cipher.decrypt(encrypted_data[AES.block_size :])
        return decrypted_data
