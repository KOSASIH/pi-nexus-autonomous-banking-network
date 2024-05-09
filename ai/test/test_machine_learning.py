import unittest
import numpy as np
from ai.machine_learning import train_linear_regression, evaluate_linear_regression

class TestMachineLearning(unittest.TestCase):
    def test_train_linear_regression(self):
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model, X_test, y_test = train_linear_regression(X, y)
        self.assertIsInstance(model, object)
        self.assertIsInstance(X_test, np.ndarray)
        self.assertIsInstance(y_test, np.ndarray)

    def test_evaluate_linear_regression(self):
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model, X_test, y_test = train_linear_regression(X, y)
        mse = evaluate_linear_regression(model, X_test, y_test)
        self.assertIsInstance(mse, float)
        self.assertGreaterEqual(mse, 0)

if __name__ == '__main__':
    unittest.main()
