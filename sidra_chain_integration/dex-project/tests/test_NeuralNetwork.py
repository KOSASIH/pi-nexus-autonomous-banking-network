import unittest
from dex.models.NeuralNetwork import NeuralNetwork
import numpy as np

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.nn = NeuralNetwork(input_shape=(10, 10), output_shape=10)

    def test_train(self):
        X_train = np.random.rand(100, 10, 10)
        y_train = np.random.rand(100, 10)
        self.nn.train(X_train, y_train, epochs=10)
        self.assertIsNotNone(self.nn.model)

    def test_predict(self):
        X_test = np.random.rand(10, 10, 10)
        predictions = self.nn.predict(X_test)
        self.assertEqual(predictions.shape, (10, 10))

    def test_save_load(self):
        self.nn.save('test_nn_model.h5')
        loaded_nn = NeuralNetwork.load('test_nn_model.h5')
        self.assertIsNotNone(loaded_nn.model)

if __name__ == '__main__':
    unittest.main()
