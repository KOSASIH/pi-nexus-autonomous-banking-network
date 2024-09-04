import hashlib
import json
from ecdsa import SigningKey, VerifyingKey
from blockchain import Blockchain, Block

class IdentityVerification:
    def __init__(self):
        self.blockchain = Blockchain()

    def generate_keys(self):
        # Generate a new pair of ECDSA keys
        signing_key = SigningKey.generate(curve=ecdsa.SECP256k1)
        verifying_key = signing_key.verifying_key
        return signing_key, verifying_key

    def create_identity(self, name, email, password):
        # Create a new identity with a unique ID
        identity_id = hashlib.sha256(name.encode() + email.encode() + password.encode()).hexdigest()
        identity = {
            'id': identity_id,
            'name': name,
            'email': email,
            'password': password,
            'public_key': self.generate_keys()[1].to_string().hex()
        }
        return identity

    def verify_identity(self, identity, signature, message):
        # Verify the identity using the public key and signature
        verifying_key = VerifyingKey.from_string(bytes.fromhex(identity['public_key']))
        return verifying_key.verify(signature, message.encode())

    def add_identity_to_blockchain(self, identity):
        # Add the identity to the blockchain
        block = Block(identity)
        self.blockchain.add_block(block)

    def get_identity_from_blockchain(self, identity_id):
        # Retrieve an identity from the blockchain
        for block in self.blockchain.chain:
            if block.data['id'] == identity_id:
                return block.data
        return None

# Example usage
identity_verification = IdentityVerification()
identity = identity_verification.create_identity('John Doe', 'johndoe@example.com', 'password123')
signature = identity_verification.generate_keys()[0].sign(b'Hello, world!')
print(identity_verification.verify_identity(identity, signature, 'Hello, world!'))  # True
identity_verification.add_identity_to_blockchain(identity)
print(identity_verification.get_identity_from_blockchain(identity['id']))  # {'id': ..., 'name': ..., 'email': ..., 'password': ..., 'public_key': ...}
