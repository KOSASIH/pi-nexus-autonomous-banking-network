import unittest
from interoperability_protocol import InteroperabilityProtocol

class TestInteroperabilityProtocol(unittest.TestCase):
    def setUp(self):
        self.protocol = InteroperabilityProtocol()

    def test_add_token(self):
        token_address = '0x...'
        self.protocol.add_token(token_address)
        self.assertIn(token_address, self.protocol.get_tokens())

    def test_remove_token(self):
        token_address = '0x...'
        self.protocol.remove_token(token_address)
        self.assertNotIn(token_address, self.protocol.get_tokens())

     def test_transfer_token(self):
        token_address = '0x...'
        from_address = '0x...'
        to_address = '0x...'
        amount = 10
        self.protocol.transfer_token(token_address, from_address, to_address, amount)
        self.assertEqual(self.protocol.get_token_balance(token_address, to_address), amount)

if __name__ == '__main__':
    unittest.main()
