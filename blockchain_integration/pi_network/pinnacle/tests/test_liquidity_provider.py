import unittest
from unittest.mock import MagicMock, patch
from pinnacle.interfaces.liquidity_provider_interface import LiquidityProviderInterface
from pinnacle.liquidity_providers.binance_liquidity_provider import BinanceLiquidityProvider

class TestLiquidityProvider(unittest.TestCase):
    def setUp(self):
        self.liquidity_provider = BinanceLiquidityProvider()

    def test_get_liquidity(self):
        # Mock the Binance API response
        response = {'balances': [{'asset': 'ETH', 'free': '10.0'}]}
        with patch('requests.get', return_value=MagicMock(json=MagicMock(return_value=response))):
            liquidity = self.liquidity_provider.get_liquidity('ETH')
            self.assertEqual(liquidity, 10.0)

    def test_place_order(self):
        # Mock the Binance API response
        response = {'orderId': '123456'}
        with patch('requests.post', return_value=MagicMock(json=MagicMock(return_value=response))):
            order_id = self.liquidity_provider.place_order('ETH', 1.0, 100.0)
            self.assertEqual(order_id, '123456')

    def test_cancel_order(self):
        # Mock the Binance API response
        response = {'status': 'CANCELED'}
        with patch('requests.delete', return_value=MagicMock(json=MagicMock(return_value=response))):
            canceled = self.liquidity_provider.cancel_order('123456')
            self.assertTrue(canceled)

    def test_get_order_book(self):
        # Mock the Binance API response
        response = {'bids': [['100.0', '1.0']], 'asks': [['101.0', '1.0']]}
        with patch('requests.get', return_value=MagicMock(json=MagicMock(return_value=response))):
            order_book = self.liquidity_provider.get_order_book('ETH')
            self.assertEqual(order_book, {'bids': [['100.0', '1.0']], 'asks': [['101.0', '1.0']]})

    def test_get_trading_pairs(self):
        # Mock the Binance API response
        response = {'symbols': ['ETHBTC', 'LTCBTC']}
        with patch('requests.get', return_value=MagicMock(json=MagicMock(return_value=response))):
            trading_pairs = self.liquidity_provider.get_trading_pairs()
            self.assertEqual(trading_pairs, ['ETHBTC', 'LTCBTC'])
