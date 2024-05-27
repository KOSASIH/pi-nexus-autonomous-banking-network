import unittest

from security import secure_generate_keypair, secure_send_transaction


class TestSecurity(unittest.TestCase):
    def test_secure_send_transaction(self):
        web3 = Web3(Web3.HTTPProvider("https://mainnet.infura.io/v3/YOUR_PROJECT_ID"))
        private_key = "0x1234567890abcdef"
        to_address = "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
        value = 1.0
        secure_send_transaction(web3, private_key, to_address, value)
        self.assertTrue(True)  # Replace with actual test logic

    def test_secure_generate_keypair(self):
        private_key, public_key = secure_generate_keypair()
        self.assertIsNotNone(private_key)
        self.assertIsNotNone(public_key)
