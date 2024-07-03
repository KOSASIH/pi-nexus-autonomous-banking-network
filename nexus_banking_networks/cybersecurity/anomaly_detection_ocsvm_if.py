# File: anomaly_detection_ocsvm_if.py
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    def __init__(self, data_path):
        self.data_path = data_path
        self.ocsvm_model = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.1)
        self.if_model = IsolationForest(contamination=0.1)

    def train(self):
        # Train one-class SVM model
        data = pd.read_csv(self.data_path)
        self.ocsvm_model.fit(data)

        # Train isolation forest model
        self.if_model.fit(data)

    def predict(self, data):
        # Predict using one-class SVM model
        ocsvm_pred = self.ocsvm_model.predict(data)

        # Predict using isolation forest model
        if_pred = self.if_model.predict(data)

        # Combine predictions using ensemble method
        pred = np.logical_or(ocsvm_pred, if_pred)
        return pred

# Example usage:
detector = AnomalyDetector('data.csv')
detector.train()
data = pd.read_csv('new_data.csv')
pred = detector.predict(data)
print(pred)
