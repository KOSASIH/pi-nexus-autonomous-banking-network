import unittest
from unittest.mock import patch
from analytics.data_ingestion import DataIngestion

class TestDataIngestion(unittest.TestCase):
    @patch('analytics.data_ingestion.DataIngestion.ingest')
    def test_ingest(self, mock_ingest):
        data_ingestion = DataIngestion("test_data_source")
        data_ingestion.ingest()
        mock_ingest.assert_called_once()

if __name__ == '__main__':
    unittest.main()
