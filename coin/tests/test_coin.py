import unittest
from .models import Coin

class TestCoin(unittest.TestCase):
    def test_coin_creation(self):
        coin = Coin(name='Test Coin', symbol='TC', value=10)
        self.assertEqual(coin.name, 'Test Coin')
        self.assertEqual(coin.symbol, 'TC')
        self.assertEqual(coin.value, 10)

if __name__ == '__main__':
    unittest.main()
