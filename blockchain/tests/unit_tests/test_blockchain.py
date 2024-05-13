import unittest
from blockchain import Blockchain

class TestBlockchain(unittest.TestCase):
    def test_add_block(self):
        """
        Test the add_block method of the Blockchain class.
        """
        blockchain = Blockchain()

        # Add a block to the blockchain
        blockchain.add_block("Transaction data")

        # Check if the block was added correctly
        self.assertEqual(len(blockchain.chain), 2)

    def test_validate_chain(self):
        """
        Test the validate_chain method of the Blockchain class.
        """
        blockchain = Blockchain()

        # Add a block to the blockchain
        blockchain.add_block("Transaction data")

        # Corrupt the chain by changing the data in one of the blocks
        blockchain.chain[1].data = "Corrupted data"

        # Validate the chain
        self.assertFalse(blockchain.validate_chain())

        # Reset the chain to its original state
        blockchain.chain[1].data = "Transaction data"

        # Validate the chain again
        self.assertTrue(blockchain.validate_chain())
