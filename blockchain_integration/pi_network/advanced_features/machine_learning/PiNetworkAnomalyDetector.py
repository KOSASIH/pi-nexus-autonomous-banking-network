# Importing necessary libraries
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Class for anomaly detection
class PiNetworkAnomalyDetector:
    def __init__(self):
        self.model = None

    # Function to train the model
    def train(self, data):
        # Training the model
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.model.fit(data)

        # Evaluating the model
        predictions = self.model.predict(data)
        print("Accuracy:", accuracy_score(data, predictions))
        print("Classification Report:")
        print(classification_report(data, predictions))
        print("Confusion Matrix:")
        print(confusion_matrix(data, predictions))

    # Function to detect anomalies
    def detect(self, data):
        # Making predictions
        predictions = self.model.predict(data)
        return predictions

# Example usage
data = pd.read_csv('anomaly_data.csv')
detector = PiNetworkAnomalyDetector()
detector.train(data)
