import numpy as np
import tensorflow as tf


class NeuralNetwork:

    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(
                    hidden_dim, activation="relu", input_shape=(input_dim,)
                ),
                tf.keras.layers.Dense(output_dim, activation="softmax"),
            ]
        )
        self.model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

    def train(self, X, y):
        self.model.fit(X, y, epochs=10, batch_size=32)

    def predict(self, X):
        return self.model.predict(X)


nn = NeuralNetwork(784, 256, 10)
X_train, y_train = ...  # load training data
nn.train(X_train, y_train)
X_test, y_test = ...  # load testing data
y_pred = nn.predict(X_test)
print(y_pred)
