import unittest
from web3 import Web3
from blockchain_utils import get_web3_provider

class TestBinanceSmartChain(unittest.TestCase):
    def setUp(self):
        self.w3 = Web3(get_web3_provider('binance_smart_chain'))

    def test_get_balance(self):
        address = '0x...'
        balance = self.w3.eth.get_balance(address)
        self.assertGreater(balance, 0)

    def test_send_transaction(self):
        from_address = '0x...'
        to_address = '0x...'
        value = 1
        tx_hash = self.w3.eth.send_transaction({'from': from_address, 'to': to_address, 'value': value})
        self.assertIsNotNone(tx_hash)

if __name__ == '__main__':
    unittest.main()
