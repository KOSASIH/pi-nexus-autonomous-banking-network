import unittest

from transaction import Transaction


class TestTransaction(unittest.TestCase):
    def test_create_transaction(self):
        transaction = Transaction("sender", "receiver", 100)
        self.assertEqual(transaction.sender, "sender")
        self.assertEqual(transaction.receiver, "receiver")
        self.assertEqual(transaction.amount, 100)
        self.assertEqual(transaction.hash, None)

    def test_hash(self):
        transaction = Transaction("sender", "receiver", 100)
        transaction.hash = None
        transaction.mine_hash()
        self.assertNotEqual(transaction.hash, None)

    def test_mine_hash(self):
        transaction = Transaction("sender", "receiver", 100)
        transaction.mine_hash(difficulty=2)
        self.assertEqual(transaction.hash[0:2], "00")


if __name__ == "__main__":
    unittest.main()
