# sidra_chain_neural_network/neural_network.py
import tensorflow as tf
from tensorflow import keras


class NeuralNetwork:
    def __init__(self):
        self.model = keras.Sequential(
            [
                keras.layers.Dense(64, activation="relu", input_shape=(10,)),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        self.model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

    def train(self, data):
        self.model.fit(data, epochs=10)

    def predict(self, data):
        return self.model.predict(data)

    def evaluate(self, data):
        return self.model.evaluate(data)
