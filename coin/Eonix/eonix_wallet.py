import hashlib
import time
import os
import json
from ecdsa import SigningKey, SECP256k1
from eonix_cryptography import eonix_encrypt, eonix_decrypt

class EonixWallet:
    def __init__(self):
        self.private_key = None
        self.public_key = None
        self.address = None
        self.wallet_file = 'eonix_wallet.json'

    def generate_keys(self):
        # Generate a new private key using SECP256k1 curve
        self.private_key = SigningKey.from_secret_exponent(1, curve=SECP256k1)
        self.public_key = self.private_key.get_verifying_key()

        # Derive the address from the public key
        self.address = hashlib.sha256(self.public_key.to_string()).hexdigest()[:20]

    def save_wallet(self):
        # Save the wallet to a file
        wallet_data = {
            'private_key': self.private_key.to_string().hex(),
            'public_key': self.public_key.to_string().hex(),
            'address': self.address
        }
        with open(self.wallet_file, 'w') as f:
            json.dump(wallet_data, f)

    def load_wallet(self):
        # Load the wallet from a file
        if os.path.exists(self.wallet_file):
            with open(self.wallet_file, 'r') as f:
                wallet_data = json.load(f)
                self.private_key = SigningKey.from_string(bytes.fromhex(wallet_data['private_key']), curve=SECP256k1)
                self.public_key = self.private_key.get_verifying_key()
                self.address = wallet_data['address']
        else:
            print("Wallet file not found. Generating new keys...")
            self.generate_keys()
            self.save_wallet()

    def sign_transaction(self, transaction):
        # Sign a transaction using the private key
        signature = self.private_key.sign(transaction.encode())
        return signature.hex()

    def encrypt_transaction(self, transaction):
        # Encrypt a transaction using Eonix's advanced encryption algorithm
        encrypted_transaction = eonix_encrypt(transaction, self.private_key)
        return encrypted_transaction

    def decrypt_transaction(self, encrypted_transaction):
        # Decrypt a transaction using Eonix's advanced decryption algorithm
        decrypted_transaction = eonix_decrypt(encrypted_transaction, self.private_key)
        return decrypted_transaction

    def get_address(self):
        return self.address

    def get_public_key(self):
        return self.public_key.to_string().hex()

    def get_private_key(self):
        return self.private_key.to_string().hex()
