import unittest
from scripts.analyze_trade_data import analyze_trade_data
from unittest.mock import patch, MagicMock
import pandas as pd

class TestAnalyzeTradeData(unittest.TestCase):
    @patch('scripts.analyze_trade_data.load_trade_data')
    @patch('scripts.analyze_trade_data.extract_trade_features')
    @patch('scripts.analyze_trade_data.TradeModel')
    def test_analyze_trade_data(self, mock_TradeModel, mock_extract_trade_features, mock_load_trade_data):
        # Mock load_trade_data to return a sample trade data
        mock_load_trade_data.return_value = pd.DataFrame({
            'date': ['2022-01-01', '2022-01-02', '2022-01-03'],
            'country': ['USA', 'Canada', 'Mexico'],
            'product': ['Apple', 'Banana', 'Cherry'],
            'quantity': [10, 20, 30],
            'value': [100, 200, 300]
        })

        # Mock extract_trade_features to return a sample trade features
        mock_extract_trade_features.return_value = pd.DataFrame({
            'date': ['2022-01-01', '2022-01-02', '2022-01-03'],
            'country': ['USA', 'Canada', 'Mexico'],
            'product': ['Apple', 'Banana', 'Cherry'],
            'quantity': [10, 20, 30],
            'value': [100, 200, 300]
        })

        # Mock TradeModel to return a sample trade model
        mock_TradeModel.return_value = MagicMock()
        mock_TradeModel.return_value.train.return_value = None
        mock_TradeModel.return_value.evaluate.return_value = 0.5

        # Call analyze_trade_data
        analyze_trade_data('data/trade-data.csv')

        # Assert that load_trade_data was called with the correct file path
        mock_load_trade_data.assert_called_with('data/trade-data.csv')

        # Assert that extract_trade_features was called with the correct trade data
        mock_extract_trade_features.assert_called_with(mock_load_trade_data.return_value)

        # Assert that TradeModel was instantiated and trained
        mock_TradeModel.assert_called_once()
        mock_TradeModel.return_value.train.assert_called_once_with(mock_extract_trade_features.return_value)

        # Assert that TradeModel was evaluated
        mock_TradeModel.return_value.evaluate.assert_called_once_with(mock_extract_trade_features.return_value)

if __name__ == '__main__':
    unittest.main()
