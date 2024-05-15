# tests/test_miner.py
import unittest
from miner import Miner
from blockchain import Blockchain
from wallet import Wallet

class TestMiner(unittest.TestCase):
    def test_mine_new_block(self):
        wallet = Wallet()
        blockchain = Blockchain(wallet)
        miner = Miner(wallet, blockchain)
        block = miner.mine_new_block([])
        self.assertIsNotNone(block)

if __name__ == '__main__':
    unittest.main()
