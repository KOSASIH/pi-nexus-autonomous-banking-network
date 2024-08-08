import unittest
from cross_chain_bridge import CrossChainBridge

class TestCrossChainBridge(unittest.TestCase):
    def setUp(self):
        self.bridge = CrossChainBridge()

    def test_lock_tokens(self):
        token_address = '0x...'
        amount = 10
        self.bridge.lock_tokens(token_address, amount)
        self.assertEqual(self.bridge.get_locked_balance(token_address), amount)

    def test_unlock_tokens(self):
        token_address = '0x...'
        amount = 10
        self.bridge.unlock_tokens(token_address, amount)
        self.assertEqual(self.bridge.get_locked_balance(token_address), 0)

if __name__ == '__main__':
    unittest.main()
