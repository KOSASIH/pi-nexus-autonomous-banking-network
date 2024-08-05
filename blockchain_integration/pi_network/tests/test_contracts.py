import unittest
from contracts import PiNetwork, PiToken

class TestPiNetworkContract(unittest.TestCase):
    def setUp(self):
        self.pi_network_contract = PiNetwork()

    def test_transfer(self):
        from_address = '0x...TestAccountAddress...'
        to_address = '0x...AnotherTestAccountAddress...'
        amount = 10
        self.pi_network_contract.transfer(from_address, to_address, amount)
        self.assertEqual(self.pi_network_contract.balances[from_address], 0)
        self.assertEqual(self.pi_network_contract.balances[to_address], amount)

    def test_balance_of(self):
        address = '0x...TestAccountAddress...'
        balance = self.pi_network_contract.balanceOf(address)
        self.assertGreater(balance, 0)

class TestPiTokenContract(unittest.TestCase):
    def setUp(self):
        self.pi_token_contract = PiToken()

    def test_transfer(self):
        from_address = '0x...TestAccountAddress...'
        to_address = '0x...AnotherTestAccountAddress...'
        amount = 10
        self.pi_token_contract.transfer(from_address, to_address, amount)
        self.assertEqual(self.pi_token_contract.balances[from_address], 0)
        self.assertEqual(self.pi_token_contract.balances[to_address], amount)

    def test_balance_of(self):
        address = '0x...TestAccountAddress...'
        balance = self.pi_token_contract.balanceOf(address)
        self.assertGreater(balance, 0)

if __name__ == '__main__':
    unittest.main()
