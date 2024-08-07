import unittest
from unittest.mock import MagicMock, patch
from pinnacle.interfaces.cross_chain_bridge_interface import CrossChainBridgeInterface
from pinnacle.cross_chain_bridges.poly_network_bridge import PolyNetworkBridge

class TestCrossChainBridge(unittest.TestCase):
    def setUp(self):
        self.cross_chain_bridge = PolyNetworkBridge()

    def test_bridge_tokens(self):
        # Mock the Poly Network API response
        response = {'txHash': '0x...'}
        with patch('requests.post', return_value=MagicMock(json=MagicMock(return_value=response))):
            tx_hash = self.cross_chain_bridge.bridge_tokens(['ETH', 'BTC'], 1.0)
            self.assertEqual(tx_hash, '0x...')

    def test_get_bridge_fee(self):
        # Mock the Poly Network API response
        response = {'fee': '0.01'}
        with patch('requests.get', return_value=MagicMock(json=MagicMock(return_value=response))):
            fee = self.cross_chain_bridge.get_bridge_fee(['ETH', 'BTC'], 1.0)
            self.assertEqual(fee, 0.01)

    def test_get_supported_chains(self):
        # Mock the Poly Network API response
        response = {'chains': ['Ethereum', 'Binance Smart Chain']}
        with patch('requests.get', return_value=MagicMock(json=MagicMock(return_value=response))):
            supported_chains = self.cross_chain_bridge.get_supported_chains()
            self.assertEqual(supported_chains, ['Ethereum', 'Binance Smart Chain'])

    def test_get_supported_tokens(self):
        # Mock the Poly Network API response
        response = {'tokens': ['ETH', 'BTC', 'LTC']}
        with patch('requests.get', return_value=MagicMock(json=MagicMock(return_value=response))):
            supported_tokens = self.cross_chain_bridge.get_supported_tokens()
            self.assertEqual(supported_tokens, ['ETH', 'BTC', 'LTC'])
