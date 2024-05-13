import unittest
from wallet import Wallet

class TestWallet(unittest.TestCase):
    def test_create_wallet(self):
        """
        Test the creation of a new wallet.
        """
        wallet = Wallet()

        # Check if the wallet has the correct attributes
        self.assertIsNotNone(wallet.address)
        self.assertIsNotNone(wallet.private_key)

    def test_add_funds(self):
        """
        Test the addition of funds to a wallet.
        """
        wallet = Wallet()

        # Add funds to the wallet
        wallet.add_funds(100)

        # Check if the wallet balance is correct
        self.assertEqual(wallet.get_balance(), 100)

    def test_send_funds(self):
        """
        Test the sending of funds from a wallet.
        """
        sender_wallet = Wallet()
        recipient_wallet = Wallet()

        # Add funds to the sender wallet
        sender_wallet.add_funds(100)

        # Send funds to the recipient wallet
        transaction = sender_wallet.send_funds(50, recipient_wallet.address)

        # Check if the transaction is valid
        self.assertTrue(transaction.validate_transaction())

        # Check if the sender wallet balance is correct
        self.assertEqual(sender_wallet.get_balance(), 50)

        # Check if the recipient wallet balance is correct
        self.assertEqual(recipient_wallet.get_balance(), 50)
