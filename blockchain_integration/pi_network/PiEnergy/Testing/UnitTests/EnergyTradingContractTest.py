import unittest
from web3 import Web3
from energy_trading_contract import EnergyTradingContract

class TestEnergyTradingContract(unittest.TestCase):
    def setUp(self):
        self.w3 = Web3(Web3.HTTPProvider("https://mainnet.infura.io/v3/YOUR_PROJECT_ID"))
        self.contract = EnergyTradingContract(self.w3, "0x742d35Cc6634C0532925a3b844Bc454e4438f44e")

    def test_deployed(self):
        self.assertTrue(self.contract.deployed())

    def test_owner(self):
        owner = self.contract.owner()
        self.assertEqual(owner, "0x1234567890abcdef")

    def test_create_trade(self):
        trade_id = self.contract.create_trade("0x1234567890abcdef", "0xabcdef1234567890", 100, 200)
        self.assertIsNotNone(trade_id)

    def test_get_trade(self):
        trade_id = 1
        trade = self.contract.get_trade(trade_id)
        self.assertIsNotNone(trade)
        self.assertEqual(trade["seller"], "0x1234567890abcdef")
        self.assertEqual(trade["buyer"], "0xabcdef1234567890")
        self.assertEqual(trade["energy_amount"], 100)
        self.assertEqual(trade["price"], 200)

    def test_execute_trade(self):
        trade_id = 1
        self.contract.execute_trade(trade_id)
        trade = self.contract.get_trade(trade_id)
        self.assertEqual(trade["status"], "EXECUTED")

    def test_cancel_trade(self):
        trade_id = 1
        self.contract.cancel_trade(trade_id)
        trade = self.contract.get_trade(trade_id)
        self.assertEqual(trade["status"], "CANCELED")

    def test_get_all_trades(self):
        trades = self.contract.get_all_trades()
        self.assertGreaterEqual(len(trades), 1)

if __name__ == "__main__":
    unittest.main()
