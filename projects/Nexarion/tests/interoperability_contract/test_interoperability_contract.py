import unittest
from web3 import Web3
from interoperability_contract import InteroperabilityContract

class TestInteroperabilityContract(unittest.TestCase):
    def setUp(self):
        self.w3 = Web3(Web3.HTTPProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'))
        self.contract = InteroperabilityContract(self.w3, '0x...')

    def test_add_token(self):
        token_address = '0x...'
        tx_hash = self.contract.functions.addToken(token_address).transact({'from': '0x...'})
        self.assertIsNotNone(tx_hash)

    def test_remove_token(self):
        token_address = '0x...'
        tx_hash = self.contract.functions.removeToken(token_address).transact({'from': '0x...'})
        self.assertIsNotNone(tx_hash)

    def test_transfer_token(self):
        token_address = '0x...'
        from_address = '0x...'
        to_address = '0x...'
        amount = 10
        tx_hash = self.contract.functions.transferToken(token_address, from_address, to_address, amount).transact({'from': '0x...'})
        self.assertIsNotNone(tx_hash)

if __name__ == '__main__':
    unittest.main()
