import unittest
from block import Block

class TestBlock(unittest.TestCase):
    def test_create_block(self):
        block = Block(1, "previous_hash", "data", 1234567890)
        self.assertEqual(block.index, 1)
        self.assertEqual(block.previous_hash, "previous_hash")
        self.assertEqual(block.data, "data")
        self.assertEqual(block.timestamp, 1234567890)
        self.assertEqual(block.hash, "hash_value")  # Replace with actual hash value

    def test_calculate_hash(self):
        block = Block(1, "previous_hash", "data", 1234567890)
        self.assertEqual(block.calculate_hash(), "hash_value")  # Replace with actual hash value

    def test_is_valid(self):
        block = Block(1, "previous_hash", "data", 1234567890)
        self.assertTrue(block.is_valid())

        # Test invalid block with modified data
        block.data = "modified_data"
        self.assertFalse(block.is_valid())

        # Test invalid block with modified previous hash
        block.previous_hash = "modified_previous_hash"
        self.assertFalse(block.is_valid())

        # Test invalid block with modified index
        block.index = 2
        self.assertFalse(block.is_valid())
