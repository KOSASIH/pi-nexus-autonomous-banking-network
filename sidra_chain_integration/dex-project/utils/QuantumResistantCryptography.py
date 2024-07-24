from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHA256

class QuantumResistantCryptography:
    def __init__(self, key):
        self.key = key
        self.cipher = AES.new(self.key, AES.MODE_GCM)

    def encrypt(self, plaintext):
        ciphertext, tag = self.cipher.encrypt_and_digest(plaintext)
        return ciphertext, tag

    def decrypt(self, ciphertext, tag):
        plaintext = self.cipher.decrypt_and_verify(ciphertext, tag)
        return plaintext
