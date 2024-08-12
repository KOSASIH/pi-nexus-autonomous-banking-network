# test_node_optimizer.py

import unittest
from node_optimizer import NodeOptimizer

class TestNodeOptimizer(unittest.TestCase):
    def setUp(self):
        self.optimizer = NodeOptimizer()

    def test_optimize_node(self):
        node = {'id': 'node-1', 'type': 'server', 'address': 'localhost', 'port': 50051}
        optimized_node = self.optimizer.optimize_node(node)
        self.assertEqual(optimized_node['id'], 'node-1')
        self.assertEqual(optimized_node['type'], 'server')
        self.assertEqual(optimized_node['address'], 'localhost')
        self.assertEqual(optimized_node['port'], 50051)
        self.assertIn('config', optimized_node)
        self.assertIn('socket', optimized_node)

    def test_optimize_node_invalid_input(self):
        with self.assertRaises(ValueError):
            self.optimizer.optimize_node({'id': 'node-1', 'type': 'invalid'})

    def test_get_optimized_nodes(self):
        nodes = [
            {'id': 'node-1', 'type': 'server', 'address': 'localhost', 'port': 50051},
            {'id': 'node-2', 'type': 'client', 'address': 'localhost', 'port': 50052},
        ]
        optimized_nodes = self.optimizer.get_optimized_nodes(nodes)
        self.assertEqual(len(optimized_nodes), 2)
        self.assertIn({'id': 'node-1', 'type': 'server', 'address': 'localhost', 'port': 50051}, optimized_nodes)
        self.assertIn({'id': 'node-2', 'type': 'client', 'address': 'localhost', 'port': 50052}, optimized_nodes)

if __name__ == '__main__':
    unittest.main()
