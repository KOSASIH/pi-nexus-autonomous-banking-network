import unittest
from unittest.mock import patch
from analytics.data_processing import DataProcessing

class TestDataProcessing(unittest.TestCase):
    @patch('analytics.data_processing.DataProcessing.process')
    def test_process(self, mock_process):
        data_processing = DataProcessing("test_data")
        data_processing.process()
        mock_process.assert_called_once()

if __name__ == '__main__':
    unittest.main()
