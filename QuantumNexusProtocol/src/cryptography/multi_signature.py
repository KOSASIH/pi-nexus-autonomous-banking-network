from ecdsa import SigningKey, VerifyingKey, SECP256k1
from hashlib import sha256

class MultiSignature:
    def __init__(self):
        self.signers = []

    def add_signer(self):
        sk = SigningKey.generate(curve=SECP256k1)
        self.signers.append(sk)

    def sign(self, message):
        signatures = []
        for sk in self.signers:
            signature = sk.sign(message)
            signatures.append(signature)
        return signatures

    def verify(self, message, signatures):
        for i, sk in enumerate(self.signers):
            vk = sk.get_verifying_key()
            if not vk.verify(signatures[i], message):
                return False
        return True

# Example usage
if __name__ == "__main__":
    multi_sig = MultiSignature()
    multi_sig.add_signer()
    multi_sig.add_signer()
    message = b"Multi-signature message"
    signatures = multi_sig.sign(message)
    is_valid = multi_sig.verify(message, signatures)
    print(f"All signatures valid: {is_valid}")
