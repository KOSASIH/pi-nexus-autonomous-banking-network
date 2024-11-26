from ecdsa import SigningKey, VerifyingKey, SECP256k1

class DigitalSignature:
    def __init__(self):
        self.sk = SigningKey.generate(curve=SECP256k1)
        self.v k = self.sk.get_verifying_key()

    def sign(self, message):
        return self.sk.sign(message)

    def verify(self, message, signature):
        return self.vk.verify(signature, message)

# Example usage
if __name__ == "__main__":
    message = b"Digital signature message"
    ds = DigitalSignature()
    signature = ds.sign(message)
    is_valid = ds.verify(message, signature)
    print(f"Signature valid: {is_valid}")
