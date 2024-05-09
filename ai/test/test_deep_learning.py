import unittest
import numpy as np
from ai.deep_learning import create_cnn_model, train_cnn_model

class TestDeepLearning(unittest.TestCase):
    def test_create_cnn_model(self):
        model = create_cnn_model()
        self.assertIsInstance(model, object)
        self.assertIsInstance(model.layers, list)

    def test_train_cnn_model(self):
        (X_train, y_train), (X_val, y_val) = (np.random.rand(100, 32, 32, 3), np.random.rand(100, 10))
        model = create_cnn_model()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = train_cnn_model(model, X_train, y_train, X_val, y_val)
        self.assertIsInstance(history, object)

if __name__ == '__main__':
    unittest.main()
