import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

class AnomalyDetection:
    def __init__(self, data):
        self.data = data

    def train_model(self):
        # Train neural network model on historical data
        X = self.data.drop(['target'], axis=1)
        y = self.data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        return model

    def detect_anomalies(self, model):
        # Detect anomalies in new data using trained model
        predictions = model.predict(self.data)
        anomalies = np.where(predictions == 1)[0]
        return anomalies
