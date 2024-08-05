import unittest
from models.market_prediction_model import MarketPredictionModel
from data_loader import load_market_data

class TestMarketPredictionModel(unittest.TestCase):
    def setUp(self):
        self.model = MarketPredictionModel()
        self.market_data = load_market_data("path/to/market-data.csv")

    def test_train(self):
        X = self.market_data.drop(["asset_id", "timestamp"], axis=1)
        y = self.market_data["price"]
        self.model.train(X, y)
        self.assertIsNotNone(self.model.model)

    def test_predict(self):
        X = self.market_data.drop(["asset_id", "timestamp"], axis=1)
        predictions = self.model.predict(X)
        self.assertIsInstance(predictions, pd.Series)

    def test_evaluate(self):
        X = self.market_data.drop(["asset_id", "timestamp"], axis=1)
        y = self.market_data["price"]
        self.model.evaluate(X, y)
        self.assertIsNotNone(self.model.metrics)

    def test_save_load(self):
        self.model.save("path/to/model.pkl")
        loaded_model = MarketPredictionModel.load("path/to/model.pkl")
        self.assertIsInstance(loaded_model, MarketPredictionModel)

if __name__ == "__main__":
    unittest.main()
