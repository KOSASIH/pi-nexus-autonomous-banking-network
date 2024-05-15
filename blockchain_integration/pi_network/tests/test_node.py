# tests/test_node.py
import unittest
from node import Node
from wallet import Wallet
from blockchain import Blockchain

class TestNode(unittest.TestCase):
    def test_join_network(self):
        wallet = Wallet()
        blockchain = Blockchain(wallet)
        node = Node(wallet, blockchain)
        node.join_network('localhost', 8080)
        self.assertIsNotNone(node.peers)

    def test_leave_network(self):
        wallet = Wallet()
        blockchain = Blockchain(wallet)
        node = Node(wallet, blockchain)
        node.join_network('localhost', 8080)
        node.leave_network()
        self.assertEqual(node.peers, [])

if __name__ == '__main__':
    unittest.main()
