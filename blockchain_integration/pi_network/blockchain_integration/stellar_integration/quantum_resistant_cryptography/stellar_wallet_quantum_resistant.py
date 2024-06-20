import hashlib
from stellar_sdk import Server, Keypair, TransactionBuilder, Network
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend

class StellarWalletQuantumResistant:
    def __init__(self, network_passphrase, horizon_url):
        self.network_passphrase = network_passphrase
        self.horizon_url = horizon_url
        self.server = Server(horizon_url)
        self.ec_curve = ec.SECP256R1()
        self.backend = default_backend()

    def generate_keypair(self, seed_phrase):
        private_key = ec.generate_private_key(self.ec_curve, self.backend)
        public_key = private_key.public_key()
        return private_key, public_key

    def create_account(self, keypair, starting_balance):
        transaction = TransactionBuilder(
            source_account=keypair.public_key,
            network_passphrase=self.network_passphrase,
            base_fee=100
        ).append_create_account_op(
            destination=keypair.public_key,
            starting_balance=starting_balance
        ).build()
        self.server.submit_transaction(transaction)

    def sign_transaction(self, private_key, transaction):
        signature = private_key.sign(transaction, padding=ec.ECDSA(max_length=hashlib.sha256().digest_size * 8 + 8))
        return signature

    def verify_signature(self, public_key, transaction, signature):
        public_key.verify(signature, transaction, padding=ec.ECDSA(max_length=hashlib.sha256().digest_size * 8 + 8))
        return True
