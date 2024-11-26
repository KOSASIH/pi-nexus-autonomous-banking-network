import unittest
from src.interoperability.cross_chain_bridge import CrossChainBridge

class TestInteroperability(unittest.TestCase):
    def setUp(self):
        self.bridge = CrossChainBridge()

    def test_token_swap(self):
        result = self.bridge.swap_tokens("tokenA", "tokenB", 10)
        self.assertTrue(result)

    def test_legacy_system_integration(self):
        result = self.bridge.integrate_legacy_system("legacy_data")
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()
