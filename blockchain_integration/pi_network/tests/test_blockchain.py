# tests/test_blockchain.py
import unittest

from blockchain import Blockchain


class TestBlockchain(unittest.TestCase):
    def test_add_block(self):
        blockchain = Blockchain()
        block = {
            "index": 1,
            "timestamp": 1643723400,
            "transactions": [],
            "previous_hash": "0" * 64,
            "nonce": 0,
        }
        blockchain.add_block(block)
        self.assertEqual(blockchain.get_latest_block(), block)

    def test_get_latest_block(self):
        blockchain = Blockchain()
        block1 = {
            "index": 1,
            "timestamp": 1643723400,
            "transactions": [],
            "previous_hash": "0" * 64,
            "nonce": 0,
        }
        block2 = {
            "index": 2,
            "timestamp": 1643723401,
            "transactions": [],
            "previous_hash": block1["hash"],
            "nonce": 0,
        }
        blockchain.add_block(block1)
        blockchain.add_block(block2)
        self.assertEqual(blockchain.get_latest_block(), block2)


if __name__ == "__main__":
    unittest.main()
