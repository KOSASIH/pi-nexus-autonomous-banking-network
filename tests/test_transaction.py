import unittest

from transaction import Transaction


class TestTransaction(unittest.TestCase):
    def test_create_transaction(self):
        transaction = Transaction("sender", "receiver", 100)
        self.assertEqual(transaction.sender, "sender")
        self.assertEqual(transaction.receiver, "receiver")
        self.assertEqual(transaction.amount, 100)
        self.assertEqual(transaction.timestamp, 1234567890)
        # Replace with actual hash value
        self.assertEqual(transaction.hash, "hash_value")

    def test_calculate_hash(self):
        transaction = Transaction("sender", "receiver", 100)
        # Replace with actual hash value
        self.assertEqual(transaction.calculate_hash(), "hash_value")

    def test_is_valid(self):
        transaction = Transaction("sender", "receiver", 100)
        self.assertTrue(transaction.is_valid())

        # Test invalid transaction with modified data
        transaction.data = "modified_data"
        self.assertFalse(transaction.is_valid())

        # Test invalid transaction with modified sender
        transaction.sender = "modified_sender"
        self.assertFalse(transaction.is_valid())

        # Test invalid transaction with modified receiver
        transaction.receiver = "modified_receiver"
        self.assertFalse(transaction.is_valid())

        # Test invalid transaction with modified amount
        transaction.amount = 200
        self.assertFalse(transaction.is_valid())

        # Test invalid transaction with modified timestamp
        transaction.timestamp = 1234567891
        self.assertFalse(transaction.is_valid())
