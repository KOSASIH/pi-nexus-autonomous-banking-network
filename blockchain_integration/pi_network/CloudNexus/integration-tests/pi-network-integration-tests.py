import unittest
from unittest.mock import patch, MagicMock
from pi_network import PiNetwork, Node, Transaction
from pi_network_client import PiNetworkClient

class TestPiNetworkIntegration(unittest.TestCase):
    def setUp(self):
        self.pi_network = PiNetwork()
        self.pi_network_client = PiNetworkClient('http://localhost:8080')

    def test_add_node(self):
        node = Node('node1', '192.168.1.100')
        response = self.pi_network_client.add_node(node)
        self.assertEqual(response.status_code, 201)
        self.assertIn(node, self.pi_network.nodes)

    def test_remove_node(self):
        node = Node('node1', '192.168.1.100')
        self.pi_network.add_node(node)
        response = self.pi_network_client.remove_node(node)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(node, self.pi_network.nodes)

    def test_add_transaction(self):
        transaction = Transaction('tx1', 'node1', 'node2', 10)
        response = self.pi_network_client.add_transaction(transaction)
        self.assertEqual(response.status_code, 201)
        self.assertIn(transaction, self.pi_network.transactions)

    def test_remove_transaction(self):
        transaction = Transaction('tx1', 'node1', 'node2', 10)
        self.pi_network.add_transaction(transaction)
        response = self.pi_network_client.remove_transaction(transaction)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(transaction, self.pi_network.transactions)

    def test_get_node_by_id(self):
        node = Node('node1', '192.168.1.100')
        self.pi_network.add_node(node)
        response = self.pi_network_client.get_node_by_id('node1')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), node.to_dict())

    def test_get_transaction_by_id(self):
        transaction = Transaction('tx1', 'node1', 'node2', 10)
        self.pi_network.add_transaction(transaction)
        response = self.pi_network_client.get_transaction_by_id('tx1')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), transaction.to_dict())

    def test_validate_transaction(self):
        transaction = Transaction('tx1', 'node1', 'node2', 10)
        self.pi_network.add_node(Node('node1', '192.168.1.100'))
        self.pi_network.add_node(Node('node2', '192.168.1.101'))
        response = self.pi_network_client.validate_transaction(transaction)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['valid'])

    def test_validate_transaction_invalid_node(self):
        transaction = Transaction('tx1', 'node1', 'node3', 10)
        self.pi_network.add_node(Node('node1', '192.168.1.100'))
        self.pi_network.add_node(Node('node2', '192.168.1.101'))
        response = self.pi_network_client.validate_transaction(transaction)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()['valid'])

    def test_validate_transaction_invalid_amount(self):
        transaction = Transaction('tx1', 'node1', 'node2', -10)
        self.pi_network.add_node(Node('node1', '192.168.1.100'))
        self.pi_network.add_node(Node('node2', '192.168.1.101'))
        response = self.pi_network_client.validate_transaction(transaction)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()['valid'])

if __name__ == '__main__':
    unittest.main()
