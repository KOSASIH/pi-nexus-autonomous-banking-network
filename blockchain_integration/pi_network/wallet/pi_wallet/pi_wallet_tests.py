import unittest

from .pi_wallet import PiWallet


class TestPiWallet(unittest.TestCase):

    def setUp(self):
        self.config = {"account_address": "test_account_address"}
        self.wallet = PiWallet(self.config)

    def test_create_account(self):
        account = self.wallet.create_account()
        self.assertIsNotNone(account.get_private_key())
        self.assertIsNotNone(account.get_public_key())

    def test_send_pi(self):
        recipient = "recipient_account_address"
        amount = 10
        transaction = self.wallet.send_pi(recipient, amount)
        self.assertIsNotNone(transaction)

    def test_receive_pi(self):
        transaction = {"amount": 10, "sender": "sender_account_address"}
        self.wallet.receive_pi(transaction)
        self.assertEqual(self.wallet.get_balance(), 10)


if __name__ == "__main__":
    unittest.main()
