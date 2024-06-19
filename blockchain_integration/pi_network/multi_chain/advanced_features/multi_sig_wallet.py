import hashlib
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

class MultiSigWallet:
    def __init__(self, chain_id, public_keys, threshold):
        self.chain_id = chain_id
        self.public_keys = public_keys
        self.threshold = threshold
        self.private_keys = {}

    def add_private_key(self, public_key, private_key):
        self.private_keys[public_key] = private_key

    def sign_transaction(self, tx_data):
        signatures = []
        for public_key, private_key in self.private_keys.items():
            signature = self.sign_with_key(tx_data, private_key)
            signatures.append(signature)
        return self.combine_signatures(signatures)

    def sign_with_key(self, tx_data, private_key):
        tx_hash = hashlib.sha256(tx_data.encode()).digest()
        signature = private_key.sign(tx_hash, ec.ECDSA(hashes.SHA256()))
        return encode_dss_signature(signature)

    def combine_signatures(self, signatures):
        combined_signature = b""
        for signature in signatures:
            combined_signature += signature
        return combined_signature

if __name__ == "__main__":
    chain_id = "ethereum"
    public_keys = [
        "0x04:1234567890abcdef",
        "0x04:fedcba9876543210",
        "0x04:0123456789abcdef"
    ]
    threshold = 2
    wallet = MultiSigWallet(chain_id, public_keys, threshold)
    wallet.add_private_key(public_keys[0], ec.generate_private_key(ec.SECP256R1()))
    wallet.add_private_key(public_keys[1], ec.generate_private_key(ec.SECP256R1()))
    tx_data = "custom transaction data"
    signature = wallet.sign_transaction(tx_data)
    print(signature)
