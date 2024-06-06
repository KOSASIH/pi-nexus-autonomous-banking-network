import pandas as pd
from sklearn.ensemble import IsolationForest

class AnomalyDetection:
    def __init__(self, data):
        self.data = data

    def detect_anomalies(self):
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(self.data)
        anomalies = model.predict(self.data)
        return anomalies
