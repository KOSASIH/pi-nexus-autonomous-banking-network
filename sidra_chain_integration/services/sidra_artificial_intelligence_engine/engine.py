# sidra_artificial_intelligence_engine/engine.py
import tensorflow as tf
import torch

class ArtificialIntelligenceEngine:
    def __init__(self):
        self.tensorflow_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation="relu", input_shape=(10,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        self.pytorch_model = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )

    def train_tensorflow_model(self, data):
        self.tensorflow_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        self.tensorflow_model.fit(data, epochs=10)

    def train_pytorch_model(self, data):
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.pytorch_model.parameters(), lr=0.001)
        for epoch in range(10):
            optimizer.zero_grad()
            outputs = self.pytorch_model(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()

    def make_predictions(self, data):
        return self.tensorflow_model.predict(data)
