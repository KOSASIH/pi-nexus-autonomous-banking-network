import unittest
from decentralized_data_analytics import DecentralizedDataAnalytics

class TestDecentralizedDataAnalytics(unittest.TestCase):
    def setUp(self):
        self.analytics = DecentralizedDataAnalytics()

    def test_get_data(self):
        data = self.analytics.get_data()
        self.assertIsNotNone(data)

    def test_analyze_data(self):
        data = self.analytics.get_data()
        result = self.analytics.analyze_data(data)
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
