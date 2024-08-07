import unittest
from unittest.mock import patch, MagicMock
from anomaly_detection import AnomalyDetector

class TestAnomalyDetector(unittest.TestCase):
    def setUp(self):
        self.anomaly_detector = AnomalyDetector()

    def test_init(self):
        self.assertIsNotNone(self.anomaly_detector)

    @patch('anomaly_detection.stats')
    def test_fit(self, mock_stats):
        mock_stats.norm.fit.return_value = (1, 2)
        data = [1, 2, 3, 4, 5]
        self.anomaly_detector.fit(data)
        mock_stats.norm.fit.assert_called_once_with(data)

    def test_predict(self):
        data = [1, 2, 3, 4, 5]
        self.anomaly_detector.fit(data)
        anomaly_score = self.anomaly_detector.predict(6)
        self.assertGreater(anomaly_score, 0.5)

    @patch('anomaly_detection.plotly')
    def test_visualize(self, mock_plotly):
        data = [1, 2, 3, 4, 5]
        self.anomaly_detector.fit(data)
        self.anomaly_detector.visualize(data)
        mock_plotly.plot.assert_called_once()

if __name__ == '__main__':
    unittest.main()
