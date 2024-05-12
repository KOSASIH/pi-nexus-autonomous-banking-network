import unittest

from node import Node


class TestNode(unittest.TestCase):
    def test_create_node(self):
        node = Node("http://localhost:5000")
        self.assertIsInstance(node.url, str)

    def test_connect_to_network(self):
        node = Node("http://localhost:5000")
        node.connect_to_network()
        self.assertGreaterEqual(len(node.network), 1)

    def test_add_transaction(self):
        node = Node("http://localhost:5000")
        node.connect_to_network()
        transaction = Transaction("sender", "receiver", 100)
        node.add_transaction(transaction)
        self.assertIn(transaction, node.pending_transactions)

    def test_mine_block(self):
        node = Node("http://localhost:5000")
        node.connect_to_network()
        node.mine_block()
        self.assertGreaterEqual(len(node.blockchain.chain), 1)

    def test_propagate_transaction(self):
        node1 = Node("http://localhost:5000")
        node2 = Node("http://localhost:5001")
        node1.connect_to_network()
        node2.connect_to_network()
        transaction = Transaction("sender", "receiver", 100)
        node1.add_transaction(transaction)
        node1.propagate_transaction(transaction)
        self.assertIn(transaction, node2.pending_transactions)

    def test_propagate_block(self):
        node1 = Node("http://localhost:5000")
        node2 = Node("http://localhost:5001")
        node1.connect_to_network()
        node2.connect_to_network()
        node1.mine_block()
        node1.propagate_block()
        self.assertGreaterEqual(len(node2.blockchain.chain), 1)
