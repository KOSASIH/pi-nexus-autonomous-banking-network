import unittest
from wallet import Wallet

class TestWallet(unittest.TestCase):
    def test_create_wallet(self):
        wallet = Wallet()
        self.assertIsInstance(wallet.private_key, str)
        self.assertIsInstance(wallet.public_key, str)
        self.assertIsInstance(wallet.address, str)

    def test_sign_transaction(self):
        wallet = Wallet()
        transaction = Transaction("sender", wallet.address, 100)
        signed_transaction = wallet.sign_transaction(transaction)
        self.assertIsInstance(signed_transaction.signature, str)

    def test_validate_signature(self):
        wallet = Wallet()
        transaction = Transaction("sender", wallet.address, 100)
        signed_transaction = wallet.sign_transaction(transaction)
        self.assertTrue(wallet.validate_signature(signed_transaction))

        # Test invalid signature with modified data
        signed_transaction.data = "modified_data"
        self.assertFalse(wallet.validate_signature(signed_transaction))

        # Test invalid signature with modified signature
        signed_transaction.signature = "modified_signature"
        self.assertFalse(wallet.validate_signature(signed_transaction))

        # Test invalid signature with modified public key
        signed_transaction.public_key = "modified_public_key"
        self.assertFalse(wallet.validate_signature(signed_transaction))
