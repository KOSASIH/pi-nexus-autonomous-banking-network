import unittest

from block import Block


class TestBlock(unittest.TestCase):
    def test_create_block(self):
        block = Block(1, "0" * 64, "test data", 100)
        self.assertEqual(block.index, 1)
        self.assertEqual(block.previous_hash, "0" * 64)
        self.assertEqual(block.data, "test data")
        self.assertEqual(block.hash, "0" * 64)
        self.assertEqual(block.nonce, 100)

    def test_hash(self):
        block = Block(1, "0" * 64, "test data", 100)
        block.hash = None
        block.mine_hash()
        self.assertNotEqual(block.hash, "0" * 64)

    def test_mine_hash(self):
        block = Block(1, "0" * 64, "test data", 100)
        block.mine_hash(difficulty=2)
        self.assertEqual(block.hash[0:2], "00")


if __name__ == "__main__":
    unittest.main()
