import unittest
from dex.models.MachineLearningModel import MachineLearningModel
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class TestMachineLearningModel(unittest.TestCase):
    def setUp(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.model = MachineLearningModel(X_train, y_train)

    def test_train(self):
        self.model.train()
        self.assertIsNotNone(self.model.model)

    def test_predict(self):
        predictions = self.model.predict(X_test)
        self.assertEqual(predictions.shape, (30,))

    def test_evaluate(self):
        accuracy = self.model.evaluate(X_test, y_test)
        self.assertGreater(accuracy, 0.5)

if __name__ == '__main__':
    unittest.main()
