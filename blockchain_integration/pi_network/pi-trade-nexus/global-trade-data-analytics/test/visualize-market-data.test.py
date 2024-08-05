import unittest
from scripts.visualize_market_data import visualize_market_data
from unittest.mock import patch, MagicMock
import pandas as pd
import matplotlib.pyplot as plt

class TestVisualizeMarketData(unittest.TestCase):
    @patch('scripts.visualize_market_data.load_market_data')
    @patch('scripts.visualize_market_data.visualize_market_data')
    def test_visualize_market_data(self, mock_visualize_market_data, mock_load_market_data):
        # Mock load_market_data to return a sample market data
        mock_load_market_data.return_value = pd.DataFrame({
            'date': ['2022-01-01', '2022-01-02', '2022-01-03'],
            'close': [100, 200, 300]
        })

        # Mock visualize_market_data to return a sample visualization
        mock_visualize_market_data.return_value = MagicMock()
        mock_visualize_market_data.return_value.show.return_value = None

        # Call visualize_market_data
        visualize_market_data('data/market-data.csv')

        # Assert that load_market_data was called with the correct file path
        mock_load_market_data.assert_called_with('data/market-data.csv')

        # Assert that visualize_market_data was called with the correct market data
        mock_visualize_market_data.assert_called_with(mock_load_market_data.return_value)

        # Assert that the visualization was shown
        mock_visualize_market_data.return_value.show.assert_called_once()

if __name__ == '__main__':
    unittest.main()
