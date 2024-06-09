import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric import utils

class PrivacyPreservingTechnologies:
    def __init__(self):
        self.rsa_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )

    def encrypt(self, message):
        ciphertext = self.rsa_key.encrypt(
            message,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return ciphertext

    def decrypt(self, ciphertext):
        plaintext = self.rsa_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return plaintext

    def sign(self, message):
        signature = self.rsa_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature

    def verify(self, message, signature):
        try:
            self.rsa_key.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except InvalidSignature:
            return False

if __name__ == '__main__':
    ppt = PrivacyPreservingTechnologies()
    message = b'Hello, Privacy-Preserving Technologies!'
    ciphertext = ppt.encrypt(message)
    print(ciphertext)

    plaintext = ppt.decrypt(ciphertext)
    print(plaintext)

    signature = ppt.sign(message)
    print(signature)

    verified = ppt.verify(message, signature)
    print(verified)
