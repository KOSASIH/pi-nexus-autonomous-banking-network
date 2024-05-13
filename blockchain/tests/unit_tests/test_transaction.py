import unittest
from blockchain.transaction import Transaction

class TestTransaction(unittest.TestCase):
    def test_create_transaction(self):
        """
        Test the creation of a new transaction.
        """
        transaction = Transaction("sender_address", "recipient_address", 100)

        # Check if the transaction has the correct attributes
        self.assertEqual(transaction.sender_address, "sender_address")
        self.assertEqual(transaction.recipient_address, "recipient_address")
        self.assertEqual(transaction.amount, 100)
        self.assertIsNotNone(transaction.timestamp)
        self.assertIsNone(transaction.hash)

    def test_hash_transaction(self):
        """
        Test the calculation of the transaction hash.
        """
        transaction = Transaction("sender_address", "recipient_address", 100)

        # Check if the transaction hash is correct
        self.assertEqual(transaction.hash, Transaction("sender_address", "recipient_address", 100).hash)

        # Change the data in the transaction
        transaction.amount = 200

        # Check if the transaction hash has changed
        self.assertNotEqual(transaction.hash, Transaction("sender_address", "recipient_address", 100).hash)
