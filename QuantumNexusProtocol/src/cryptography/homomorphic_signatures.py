from phe import paillier

class HomomorphicSignatures:
    def __init__(self):
        self.public_key, self.private_key = paillier.generate_paillier_keypair()

    def sign(self, value):
        return self.private_key.encrypt(value)

    def verify(self, encrypted_value, signature):
        return self.public_key.decrypt(signature) == encrypted_value

# Example usage
if __name__ == "__main__":
    hs = HomomorphicSignatures()
    value = 42
    signature = hs.sign(value)
    is_valid = hs.verify(value, signature)
    print(f"Signature valid: {is_valid}")
