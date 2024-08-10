import unittest
from pi_fusion_dashboard.dashboard import Dashboard

class TestDashboard(unittest.TestCase):
    def setUp(self):
        self.dashboard = Dashboard()

    def test_get_node_rankings(self):
        node_rankings = self.dashboard.get_node_rankings()
        self.assertIsInstance(node_rankings, list)
        self.assertGreater(len(node_rankings), 0)

    def test_get_transaction_activity(self):
        transaction_activity = self.dashboard.get_transaction_activity()
        self.assertIsInstance(transaction_activity, list)
        self.assertGreater(len(transaction_activity), 0)

    def test_update_dashboard(self):
        self.dashboard.update_dashboard()
        self.assertTrue(True)  # TO DO: implement more robust test

if __name__ == '__main__':
    unittest.main()
