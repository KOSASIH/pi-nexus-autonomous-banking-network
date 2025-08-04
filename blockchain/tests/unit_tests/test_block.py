import unittest

from blockchain.block import Block


class TestBlock(unittest.TestCase):
    def test_create_block(self):
        """
        Test the creation of a new block.
        """
        block = Block(1, "previous_hash", "data")

        # Check if the block has the correct attributes
        self.assertEqual(block.index, 1)
        self.assertEqual(block.previous_hash, "previous_hash")
        self.assertEqual(block.data, "data")
        self.assertIsNotNone(block.timestamp)
        self.assertIsNotNone(block.hash)

    def test_hash_block(self):
        """
        Test the calculation of the block hash.
        """
        block = Block(1, "previous_hash", "data")

        # Change the data in the block
        block.data = "new_data"

        # Check if the block hash has changed
        self.assertNotEqual(block.hash, Block(1, "previous_hash", "data").hash)

    def test_validate_block(self):
        """
        Test the validation of a block.
        """
        block = Block(1, "previous_hash", "data")

        # Check if the block is valid
        self.assertTrue(block.validate_block())

        # Change the data in the block
        block.data = "new_data"

        # Check if the block is invalid
        self.assertFalse(block.validate_block())
