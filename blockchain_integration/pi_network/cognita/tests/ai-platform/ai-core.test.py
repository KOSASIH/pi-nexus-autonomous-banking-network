import unittest
from ai_platform.src.ai_core.ai_engine import AIEngine

class TestAIEngine(unittest.TestCase):
    def test_train(self):
        # Test training logic
        engine = AIEngine()
        engine.train(X_train, y_train)
        self.assertEqual(engine.model.layers[0].get_weights()[0].shape, (784, 64))

    def test_predict(self):
        # Test prediction logic
        engine = AIEngine()
        predictions = engine.predict(X_test)
        self.assertEqual(predictions.shape, (10,))

if __name__ == '__main__':
    unittest.main()
