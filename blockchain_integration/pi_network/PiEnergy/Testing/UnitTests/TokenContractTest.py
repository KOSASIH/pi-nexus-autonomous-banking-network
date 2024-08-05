import unittest
from web3 import Web3
from token_contract import TokenContract

class TestTokenContract(unittest.TestCase):
    def setUp(self):
        self.w3 = Web3(Web3.HTTPProvider("https://mainnet.infura.io/v3/YOUR_PROJECT_ID"))
        self.contract = TokenContract(self.w3, "0x742d35Cc6634C0532925a3b844Bc454e4438f44e")

    def test_deployed(self):
        self.assertTrue(self.contract.deployed())

    def test_name(self):
        name = self.contract.name()
        self.assertEqual(name, "EnergyToken")

    def test_symbol(self):
        symbol = self.contract.symbol()
        self.assertEqual(symbol, "ETK")

    def test_total_supply(self):
        total_supply = self.contract.total_supply()
        self.assertGreaterEqual(total_supply, 1000000)

    def test_balance_of(self):
        address = "0x1234567890abcdef"
        balance = self.contract.balance_of(address)
        self.assertGreaterEqual(balance, 100)

    def test_transfer(self):
        sender = "0x1234567890abcdef"
        recipient = "0xabcdef1234567890"
        amount = 100
        self.contract.transfer(sender, recipient, amount)
        balance_sender = self.contract.balance_of(sender)
        balance_recipient = self.contract.balance_of(recipient)
        self.assertEqual(balance_sender, 0)
        self.assertEqual(balance_recipient, amount)

    def test_approve(self):
        owner = "0x1234567890abcdef"
        spender = "0xabcdef1234567890"
        amount = 100
        self.contract.approve(owner, spender, amount)
        allowance = self.contract.allowance(owner, spender)
        self.assertEqual(allowance, amount)

    def test_transfer_from(self):
        sender = "0x1234567890abcdef"
        recipient = "0xabcdef1234567890"
        amount = 100
        self.contract.transfer_from(sender, recipient, amount)
        balance_sender = self.contract.balance_of(sender)
        balance_recipient = self.contract.balance_of(recipient)
        self.assertEqual(balance_sender, 0)
        self.assertEqual(balance_recipient, amount)

if __name__ == "__main__":
    unittest.main()
