import unittest

from blockchain import Blockchain


class TestBlockchain(unittest.TestCase):
    def setUp(self):
        self.blockchain = Blockchain()

    def test_create_genesis_block(self):
        self.assertEqual(self.blockchain.chain[0].index, 0)
        self.assertEqual(self.blockchain.chain[0].previous_hash, "0" * 64)
        self.assertEqual(self.blockchain.chain[0].data, "Genesis Block")
        self.assertEqual(self.blockchain.chain[0].timestamp, 1234567890)
        # Replace with actual hash value
        self.assertEqual(self.blockchain.chain[0].hash, "hash_value")

    def test_add_block(self):
        self.blockchain.add_block("data")
        self.assertEqual(len(self.blockchain.chain), 2)
        self.assertEqual(self.blockchain.chain[1].index, 1)
        self.assertEqual(
            self.blockchain.chain[1].previous_hash, self.blockchain.chain[0].hash
        )
        self.assertEqual(self.blockchain.chain[1].data, "data")
        self.assertEqual(self.blockchain.chain[1].timestamp, 1234567891)
        # Replace with actual hash value
        self.assertEqual(self.blockchain.chain[1].hash, "hash_value")

    def test_is_valid(self):
        self.assertTrue(self.blockchain.is_valid())

        # Test invalid blockchain with modified data
        self.blockchain.chain[1].data = "modified_data"
        self.assertFalse(self.blockchain.is_valid())

        # Test invalid blockchain with modified previous hash
        self.blockchain.chain[1].previous_hash = "modified_previous_hash"
        self.assertFalse(self.blockchain.is_valid())

        # Test invalid blockchain with modified index
        self.blockchain.chain[1].index = 2
        self.assertFalse(self.blockchain.is_valid())

        # Test invalid blockchain with modified genesis block
        self.blockchain.chain[0].data = "modified_genesis_block"
        self.assertFalse(self.blockchain.is_valid())
