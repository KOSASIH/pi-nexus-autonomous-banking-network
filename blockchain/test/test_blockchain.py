import unittest

from blockchain import Blockchain


class TestBlockchain(unittest.TestCase):
    def setUp(self):
        self.blockchain = Blockchain()

    def test_create_genesis_block(self):
        genesis_block = self.blockchain.create_genesis_block()
        self.assertEqual(genesis_block.index, 0)
        self.assertEqual(genesis_block.previous_hash, "0" * 64)
        self.assertIsNotNone(genesis_block.timestamp)
        self.assertEqual(genesis_block.data, "Genesis Block")
        self.assertEqual(genesis_block.hash, "0" * 64)

    def test_calculate_hash(self):
        block = self.blockchain.create_genesis_block()
        block.data = "New Data"
        block.hash = None
        calculated_hash = self.blockchain.calculate_hash(
            block.index, block.previous_hash, block.timestamp, block.data
        )
        self.assertEqual(calculated_hash, "0" * 64)

    def test_is_chain_valid(self):
        self.assertTrue(self.blockchain.is_chain_valid(self.blockchain.chain))

        invalid_chain = self.blockchain.chain[:-1]
        invalid_chain.append(
            Block(
                len(invalid_chain),
                invalid_chain[-1].hash,
                int(time.time()),
                "Invalid Data",
                None,
            )
        )
        self.assertFalse(self.blockchain.is_chain_valid(invalid_chain))

    def test_add_block(self):
        self.blockchain.add_block("Test Data")
        self.assertEqual(len(self.blockchain.chain), 2)
        self.assertEqual(self.blockchain.chain[-1].data, "Test Data")

    def test_replace_chain(self):
        self.blockchain.replace_chain(self.blockchain.chain[:-1])
        self.assertEqual(len(self.blockchain.chain), 1)

        invalid_chain = self.blockchain.chain[:-1]
        invalid_chain.append(
            Block(
                len(invalid_chain),
                invalid_chain[-1].hash,
                int(time.time()),
                "Invalid Data",
                None,
            )
        )
        self.blockchain.replace_chain(invalid_chain)
        self.assertFalse(self.blockchain.is_chain_valid(self.blockchain.chain))

        valid_chain = self.blockchain.chain[:-1]
        valid_chain.append(
            Block(
                len(valid_chain),
                valid_chain[-1].hash,
                int(time.time()),
                "Valid Data",
                None,
            )
        )
        self.blockchain.replace_chain(valid_chain)
        self.assertTrue(self.blockchain.is_chain_valid(self.blockchain.chain))
