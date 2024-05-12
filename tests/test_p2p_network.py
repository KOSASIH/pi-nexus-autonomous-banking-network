import unittest

from p2p_network import P2PNetwork


class TestP2PNetwork(unittest.TestCase):
    def test_create_network(self):
        network = P2PNetwork()
        self.assertIsInstance(network.nodes, set)

    def test_add_node(self):
        network = P2PNetwork()
        node = Node("http://localhost:5000")
        network.add_node(node)
        self.assertIn(node, network.nodes)

    def test_remove_node(self):
        network = P2PNetwork()
        node = Node("http://localhost:5000")
        network.add_node(node)
        network.remove_node(node)
        self.assertNotIn(node, network.nodes)

    def test_connect_nodes(self):
        network = P2PNetwork()
        node1 = Node("http://localhost:5000")
        node2 = Node("http://localhost:5001")
        network.add_node(node1)
        network.add_node(node2)
        network.connect_nodes()
        self.assertIn(node2, node1.network)
        self.assertIn(node1, node2.network)

    def test_disconnect_nodes(self):
        network = P2PNetwork()
        node1 = Node("http://localhost:5000")
        node2 = Node("http://localhost:5001")
        network.add_node(node1)
        network.add_node(node2)
        network.connect_nodes()
        network.disconnect_nodes()
        self.assertNotIn(node2, node1.network)
        self.assertNotIn(node1, node2.network)

    def test_propagate_transaction(self):
        network = P2PNetwork()
        node1 = Node("http://localhost:5000")
        node2 = Node("http://localhost:5001")
        network.add_node(node1)
        network.add_node(node2)
        network.connect_nodes()
        transaction = Transaction("sender", "receiver", 100)
        node1.add_transaction(transaction)
        network.propagate_transaction(transaction)
        self.assertIn(transaction, node2.pending_transactions)

    def test_propagate_block(self):
        network = P2PNetwork()
        node1 = Node("http://localhost:5000")
        node2 = Node("http://localhost:5001")
        network.add_node(node1)
        network.add_node(node2)
        network.connect_nodes()
        node1.mine_block()
        network.propagate_block()
        self.assertGreaterEqual(len(node2.blockchain.chain), 1)
