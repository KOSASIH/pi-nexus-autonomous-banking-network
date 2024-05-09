import unittest
from unittest.mock import patch
from analytics.data_visualization import DataVisualization

class TestDataVisualization(unittest.TestCase):
    @patch('analytics.data_visualization.DataVisualization.visualize')
    def test_visualize(self, mock_visualize):
        data_visualization = DataVisualization("test_data")
        data_visualization.visualize()
        mock_visualize.assert_called_once()

if __name__ == '__main__':
    unittest.main()
