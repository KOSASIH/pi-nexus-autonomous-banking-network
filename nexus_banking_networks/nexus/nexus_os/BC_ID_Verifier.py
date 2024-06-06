import hashlib
from ecdsa import SigningKey, VerifyingKey

class BCIDVerifier:
    def __init__(self, blockchain):
        self.blockchain = blockchain

   def generate_keypair(self):
        sk = SigningKey.from_secret_exponent(123, curve=hashlib.sha256)
        vk = sk.verifying_key
        return sk, vk

    def verify_identity(self, public_key, signature, message):
        vk = VerifyingKey.from_string(public_key)
        return vk.verify(signature, message)

# Example usage:
bc_id_verifier = BCIDVerifier('nexus_blockchain')
sk, vk = bc_id_verifier.generate_keypair()

message = 'Hello, Nexus OS!'
signature = sk.sign(message.encode())
print(f'Signature: {signature}')

verified = bc_id_verifier.verify_identity(vk.to_string(), signature, message)
print(f'Verified: {verified}')
