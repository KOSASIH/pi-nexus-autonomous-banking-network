import unittest
from ai_liquidity_aggregator import AILiquidityAggregator

class TestAILiquidityAggregator(unittest.TestCase):
    def setUp(self):
        self.aggregator = AILiquidityAggregator()

    def test_get_liquidity(self):
        token_address = '0x...'
        liquidity = self.aggregator.get_liquidity(token_address)
        self.assertGreater(liquidity, 0)

    def test_execute_trade(self):
        token_address = '0x...'
        amount = 10
        self.aggregator.execute_trade(token_address, amount)
        self.assertEqual(self.aggregator.get_liquidity(token_address), amount)

if __name__ == '__main__':
    unittest.main()
