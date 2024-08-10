import unittest
from pi_fusion_dashboard.node_selection import NodeSelection

class TestNodeSelection(unittest.TestCase):
    def setUp(self):
        self.node_selection = NodeSelection()

    def test_get_nodes(self):
        nodes = self.node_selection.get_nodes()
        self.assertIsInstance(nodes, list)
        self.assertGreater(len(nodes), 0)

    def test_get_node_by_id(self):
        node_id = 1
        node = self.node_selection.get_node_by_id(node_id)
        self.assertIsInstance(node, dict)
        self.assertEqual(node['id'], node_id)

    def test_get_node_by_name(self):
        node_name = 'Node 1'
        node = self.node_selection.get_node_by_name(node_name)
        self.assertIsInstance(node, dict)
        self.assertEqual(node['name'], node_name)

    def test_filter_nodes_by_reputation(self):
        reputation_threshold = 0.5
        filtered_nodes = self.node_selection.filter_nodes_by_reputation(reputation_threshold)
        self.assertIsInstance(filtered_nodes, list)
        for node in filtered_nodes:
            self.assertGreaterEqual(node['reputation'], reputation_threshold)

if __name__ == '__main__':
    unittest.main()
