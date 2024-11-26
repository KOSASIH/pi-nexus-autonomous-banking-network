from pqcrypto.kem import saber

class PostQuantumCrypto:
    def __init__(self):
        self.public_key, self.private_key = saber.generate_keypair()

    def encrypt(self, message):
        ciphertext, shared_secret = saber.encrypt(self.public_key, message)
        return ciphertext, shared_secret

    def decrypt(self, ciphertext):
        message, shared_secret = saber.decrypt(ciphertext, self.private_key)
        return message

# Example usage
if __name__ == "__main__":
    pq_crypto = PostQuantumCrypto()
    message = b"Hello, Quantum World!"
    ciphertext, shared_secret = pq_crypto.encrypt(message)
    print(f"Ciphertext: {ciphertext}")
    decrypted_message = pq_crypto.decrypt(ciphertext)
        print(f"Decrypted Message: {decrypted_message}")
