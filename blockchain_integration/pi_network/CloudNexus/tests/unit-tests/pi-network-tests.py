import unittest
from unittest.mock import patch, MagicMock
from pi_network import PiNetwork, Node, Transaction

class TestPiNetwork(unittest.TestCase):
    def setUp(self):
        self.pi_network = PiNetwork()

    def test_init(self):
        self.assertIsInstance(self.pi_network, PiNetwork)
        self.assertEqual(self.pi_network.nodes, [])
        self.assertEqual(self.pi_network.transactions, [])

    def test_add_node(self):
        node = Node('node1', '192.168.1.100')
        self.pi_network.add_node(node)
        self.assertIn(node, self.pi_network.nodes)

    def test_remove_node(self):
        node = Node('node1', '192.168.1.100')
        self.pi_network.add_node(node)
        self.pi_network.remove_node(node)
        self.assertNotIn(node, self.pi_network.nodes)

    def test_add_transaction(self):
        transaction = Transaction('tx1', 'node1', 'node2', 10)
        self.pi_network.add_transaction(transaction)
        self.assertIn(transaction, self.pi_network.transactions)

    def test_remove_transaction(self):
        transaction = Transaction('tx1', 'node1', 'node2', 10)
        self.pi_network.add_transaction(transaction)
        self.pi_network.remove_transaction(transaction)
        self.assertNotIn(transaction, self.pi_network.transactions)

    @patch('pi_network.Node')
    def test_get_node_by_id(self, mock_node):
        node = Node('node1', '192.168.1.100')
        mock_node.return_value = node
        self.pi_network.add_node(node)
        self.assertEqual(self.pi_network.get_node_by_id('node1'), node)

    @patch('pi_network.Transaction')
    def test_get_transaction_by_id(self, mock_transaction):
        transaction = Transaction('tx1', 'node1', 'node2', 10)
        mock_transaction.return_value = transaction
        self.pi_network.add_transaction(transaction)
        self.assertEqual(self.pi_network.get_transaction_by_id('tx1'), transaction)

    def test_validate_transaction(self):
        transaction = Transaction('tx1', 'node1', 'node2', 10)
        self.pi_network.add_node(Node('node1', '192.168.1.100'))
        self.pi_network.add_node(Node('node2', '192.168.1.101'))
        self.assertTrue(self.pi_network.validate_transaction(transaction))

    def test_validate_transaction_invalid_node(self):
        transaction = Transaction('tx1', 'node1', 'node3', 10)
        self.pi_network.add_node(Node('node1', '192.168.1.100'))
        self.pi_network.add_node(Node('node2', '192.168.1.101'))
        self.assertFalse(self.pi_network.validate_transaction(transaction))

    def test_validate_transaction_invalid_amount(self):
        transaction = Transaction('tx1', 'node1', 'node2', -10)
        self.pi_network.add_node(Node('node1', '192.168.1.100'))
        self.pi_network.add_node(Node('node2', '192.168.1.101'))
        self.assertFalse(self.pi_network.validate_transaction(transaction))

if __name__ == '__main__':
    unittest.main()
