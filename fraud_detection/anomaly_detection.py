import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

class AnomalyDetection:
    def __init__(self, transaction_data):
        self.transaction_data = transaction_data

    def detect_anomalies(self):
        clf = IsolationForest(random_state=42)
        clf.fit(self.transaction_data)
        self.anomalies = clf.predict(self.transaction_data)

    def get_anomalies(self):
        return self.anomalies
