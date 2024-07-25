import pandas as pd
from sklearn.ensemble import IsolationForest

class PiNetworkAnomalyDetection:
    def __init__(self):
        self.model = IsolationForest()

    def detect(self, user):
        # Load user data
        user_data = pd.read_csv(f"user_data/{user}.csv")

        # Detect anomaly using AI model
        anomaly_score = self.model.decision_function(user_data)

        return anomaly_score
