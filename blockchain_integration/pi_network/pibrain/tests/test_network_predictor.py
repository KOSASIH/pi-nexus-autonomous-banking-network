# test_network_predictor.py

import unittest
from network_predictor import NetworkPredictor

class TestNetworkPredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = NetworkPredictor()

    def test_predict_network_traffic(self):
        traffic_data = [
            {'timestamp': 1643723400, 'bytes_sent': 1000, 'bytes_received': 500},
            {'timestamp': 1643723410, 'bytes_sent': 2000, 'bytes_received': 1000},
            {'timestamp': 1643723420, 'bytes_sent': 3000, 'bytes_received': 1500},
        ]
        prediction = self.predictor.predict_network_traffic(traffic_data)
        self.assertIsInstance(prediction, dict)
        self.assertIn('predicted_bytes_sent', prediction)
        self.assertIn('predicted_bytes_received', prediction)

    def test_predict_network_traffic_invalid_input(self):
        with self.assertRaises(ValueError):
            self.predictor.predict_network_traffic([{'timestamp': 1643723400, 'bytes_sent': 'invalid'}])

    def test_get_network_traffic_forecast(self):
        traffic_data = [
            {'timestamp': 1643723400, 'bytes_sent': 1000, 'bytes_received': 500},
            {'timestamp': 1643723410, 'bytes_sent': 2000, 'bytes_received': 1000},
            {'timestamp': 1643723420, 'bytes_sent': 3000, 'bytes_received': 1500},
        ]
        forecast = self.predictor.get_network_traffic_forecast(traffic_data, 3)
        self.assertIsInstance(forecast, list)
        self.assertEqual(len(forecast), 3)
        for item in forecast:
            self.assertIsInstance(item, dict)
            self.assertIn('timestamp', item)
            self.assertIn('predicted_bytes_sent', item)
            self.assertIn('predicted_bytes_received', item)

if __name__ == '__main__':
    unittest.main()
