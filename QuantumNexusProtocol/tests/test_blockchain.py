import unittest
from src.core.blockchain import Blockchain

class TestBlockchain(unittest.TestCase):
    def setUp(self):
        self.blockchain = Blockchain()

    def test_initial_block(self):
        self.assertEqual(len(self.blockchain.chain), 1)
        self.assertEqual(self.blockchain.chain[0]['index'], 1)

    def test_add_block(self):
        previous_length = len(self.blockchain.chain)
        self.blockchain.add_block(data="Test Block")
        self.assertEqual(len(self.blockchain.chain), previous_length + 1)

    def test_chain_integrity(self):
        self.blockchain.add_block(data="Block 1")
        self.blockchain.add_block(data="Block 2")
        self.assertTrue(self.blockchain.is_chain_valid())

if __name__ == '__main__':
    unittest.main()
