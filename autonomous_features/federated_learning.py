# federated_learning.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class FederatedLearning:
    def __init__(self):
        self.model = self.create_model()

    def create_model(self):
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(10,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, data):
        self.model.fit(data, epochs=10, batch_size=32)

    def evaluate_model(self, data):
        loss, accuracy = self.model.evaluate(data)
        return loss, accuracy

    def federated_train(self, data, clients):
        for client inclients:
            client_data = data[client]
            self.train_model(client_data)
            self.model.fit(client_data, epochs=10, batch_size=32)
        return self.model
