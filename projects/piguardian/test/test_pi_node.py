import unittest
from unittest.mock import patch, MagicMock
from pi_node import PiNode

class TestPiNode(unittest.TestCase):
    def setUp(self):
        self.pi_node = PiNode()

    def test_init(self):
        self.assertIsNotNone(self.pi_node)

    @patch('pi_node.requests')
    def test_send_data(self, mock_requests):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_requests.post.return_value = mock_response

        data = {'temperature': 25, 'humidity': 60}
        self.pi_node.send_data(data)

        mock_requests.post.assert_called_once_with('https://example.com/api/data', json=data)

    @patch('pi_node.time')
    def test_read_sensor(self, mock_time):
        mock_time.sleep.return_value = None
        mock_sensor_data = {'temperature': 25, 'humidity': 60}

        with patch('pi_node.DHTSensor') as mock_dht_sensor:
            mock_dht_sensor.return_value.read.return_value = mock_sensor_data
            data = self.pi_node.read_sensor()
            self.assertEqual(data, mock_sensor_data)

    def test_calculate_pi(self):
        data = {'temperature': 25, 'humidity': 60}
        pi_value = self.pi_node.calculate_pi(data)
        self.assertAlmostEqual(pi_value, 3.14159, places=5)

if __name__ == '__main__':
    unittest.main()
