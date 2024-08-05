import unittest
from models.trade_matching_model import TradeMatchingModel
from data_loader import load_trade_data

class TestTradeMatchingModel(unittest.TestCase):
    def setUp(self):
        self.model = TradeMatchingModel()
        self.trade_data = load_trade_data("path/to/trade-data.csv")

    def test_train(self):
        X = self.trade_data.drop(["trade_id", "timestamp"], axis=1)
        y = self.trade_data["trade_id"]
        self.model.train(X, y)
        self.assertIsNotNone(self.model.model)

    def test_predict(self):
        X = self.trade_data.drop(["trade_id", "timestamp"], axis=1)
        predictions = self.model.predict(X)
        self.assertIsInstance(predictions, pd.Series)

    def test_evaluate(self):
        X = self.trade_data.drop(["trade_id", "timestamp"], axis=1)
        y = self.trade_data["trade_id"]
        self.model.evaluate(X, y)
        self.assertIsNotNone(self.model.metrics)

    def test_save_load(self):
        self.model.save("path/to/model.pkl")
        loaded_model = TradeMatchingModel.load("path/to/model.pkl")
        self.assertIsInstance(loaded_model, TradeMatchingModel)

if __name__ == "__main__":
    unittest.main()
