import hashlib

import ecdsa
from ecdsa.keys import VerifyingKey
from ecdsa.util import sigdecode_der


class MultiSigWallet:
    def __init__(self, num_signatures_required, num_signatures_total):
        self.num_signatures_required = num_signatures_required
        self.num_signatures_total = num_signatures_total
        self.public_keys = []
        self.private_keys = []

    def generate_keys(self):
        for _ in range(self.num_signatures_total):
            private_key = ecdsa.SigningKey.from_secret_exponent(
                123, curve=ecdsa.SECP256k1
            )
            public_key = private_key.get_verifying_key()
            self.private_keys.append(private_key)
            self.public_keys.append(public_key)

    def create_multisig_address(self):
        # Create a multisig address using the public keys
        multisig_address = hashlib.sha256(
            b"".join([pk.to_string() for pk in self.public_keys])
        ).hexdigest()
        return multisig_address

    def sign_transaction(self, transaction, private_key_index):
        private_key = self.private_keys[private_key_index]
        signature = private_key.sign(transaction)
        return signature

    def verify_signature(self, signature, public_key_index, transaction):
        public_key = self.public_keys[public_key_index]
        try:
            public_key.verify(signature, transaction)
            return True
        except ecdsa.BadSignatureError:
            return False

    def authorize_transaction(self, transaction, signatures):
        if len(signatures) < self.num_signatures_required:
            return False
        for signature in signatures:
            public_key_index = self.public_keys.index(
                VerifyingKey.from_string(signature)
            )
            if not self.verify_signature(signature, public_key_index, transaction):
                return False
        return True


# Example usage:
wallet = MultiSigWallet(2, 3)  # 2 signatures required, 3 total signatures
wallet.generate_keys()
multisig_address = wallet.create_multisig_address()
print(multisig_address)

transaction = b"Hello, world!"
signature1 = wallet.sign_transaction(transaction, 0)
signature2 = wallet.sign_transaction(transaction, 1)

signatures = [signature1, signature2]
if wallet.authorize_transaction(transaction, signatures):
    print("Transaction authorized!")
else:
    print("Transaction not authorized!")
