import unittest
from src.ai.predictive_analytics import PredictiveAnalytics

class TestAIFeatures(unittest.TestCase):
    def setUp(self):
        self.ai = PredictiveAnalytics()

    def test_prediction_accuracy(self):
        accuracy = self.ai.predict_outcome(data=[1, 2, 3])
        self.assertGreaterEqual(accuracy, 0.8)  # Assuming 80% accuracy is the threshold

if __name__ == '__main__':
    unittest.main()
