import unittest
from unittest.mock import patch
from analytics.data_analysis import DataAnalysis

class TestDataAnalysis(unittest.TestCase):
    @patch('analytics.data_analysis.DataAnalysis.analyze')
    def test_analyze(self, mock_analyze):
        data_analysis = DataAnalysis("test_data")
        data_analysis.analyze()
        mock_analyze.assert_called_once()

if __name__ == '__main__':
    unittest.main()
