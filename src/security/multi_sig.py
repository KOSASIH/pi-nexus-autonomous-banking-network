import hashlib
import ecdsa

class MultiSig:
    def __init__(self, num_signers):
        self.num_signers = num_signers
        self.signers = [ecdsa.SigningKey.from_secret_exponent(i, curve=ecdsa.SECP256k1) for i in range(1, num_signers + 1)]

    def sign(self, message, signer_index):
        signer = self.signers[signer_index - 1]
        signature = signer.sign(hashlib.sha256(message.encode()).digest())
        return signature

    def verify(self, message, signature, signer_index):
        signer = self.signers[signer_index - 1]
        verifying_key = signer.get_verifying_key()
        try:
            verifying_key.verify(signature, hashlib.sha256(message.encode()).digest())
            return True
        except ecdsa.BadSignatureError:
            return False

    @staticmethod
    def combine_signatures(signatures):
        combined_signature = b''
        for signature in signatures:
            combined_signature += signature
        return combined_signature

    def verify_combined(self, message, combined_signature):
        for i in range(self.num_signers):
            signer = self.signers[i]
            verifying_key = signer.get_verifying_key()
            try:
                verifying_key.verify(combined_signature, hashlib.sha256(message.encode()).digest())
            except ecdsa.BadSignatureError:
                return False
        return True

# Example usage
if __name__ == "__main__":
    multi_sig = MultiSig(num_signers=3)
    message = "Hello, Multi-Sig!"
    signatures = [multi_sig.sign(message, i) for i in range(1, 4)]
    combined_signature = multi_sig.combine_signatures(signatures)
    print(multi_sig.verify_combined(message, combined_signature))
