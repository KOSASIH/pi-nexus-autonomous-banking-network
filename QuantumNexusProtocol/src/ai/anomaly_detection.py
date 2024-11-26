import numpy as np
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    def __init__(self, data):
        self.data = data
        self.model = IsolationForest()

    def detect_anomalies(self):
        self.model.fit(self.data)
        anomalies = self.model.predict(self.data)
        return anomalies

# Example usage
if __name__ == "__main__":
    data = np.random.rand(100, 5)  # Simulated data
    detector = AnomalyDetector(data)
    anomalies = detector.detect_anomalies()
    print("Anomalies Detected:", anomalies)
