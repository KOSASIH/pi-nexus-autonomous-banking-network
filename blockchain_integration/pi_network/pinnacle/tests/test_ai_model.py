import unittest
from unittest.mock import MagicMock, patch
from pinnacle.utils.ai_utils import AIUtils
from pinnacle.ai_models.prophet_model import ProphetModel

class TestAIModel(unittest.TestCase):
    def setUp(self):
        self.ai_utils = AIUtils(ai_model=ProphetModel())

    def test_make_predictions(self):
        # Mock the Prophet model
        prophet_model = MagicMock()
        prophet_model.make_future_dataframe.return_value = 'future_df'
        prophet_model.predict.return_value = 'predictions'
        with patch('pinnacle.ai_models.prophet_model.Prophet', return_value=prophet_model):
            predictions = self.ai_utils.make_predictions('ETH', 30)
            self.assertEqual(predictions, 'predictions')

    def test_train_model(self):
        # Mock the Prophet model
        prophet_model = MagicMock()
        prophet_model.fit.return_value = None
        with patch('pinnacle.ai_models.prophet_model.Prophet', return_value=prophet_model):
            self.ai_utils.train_model('ETH', 'data.csv')

    def test_get_model_performance(self):
        # Mock the Prophet model
        prophet_model = MagicMock()
        prophet_model.evaluate.return_value = {'mse': 0.01}
        with patch('pinnacle.ai_models.prophet_model.Prophet', return_value=prophet_model):
            performance = self.ai_utils.get_model_performance('ETH')
            self.assertEqual(performance, {'mse': 0.01})

if __name__ == '__main__':
    unittest.main()
