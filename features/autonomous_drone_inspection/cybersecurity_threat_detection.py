# File name: cybersecurity_threat_detection.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class ThreatDetector:
    def __init__(self, model_path):
        self.model = RandomForestClassifier(n_estimators=100)
        self.model.load_model(model_path)

    def detect_threats(self, data):
        X = pd.DataFrame(data, columns=["feature1", "feature2", "feature3"])
        y_pred = self.model.predict(X)
        return y_pred


threat_detector = ThreatDetector("threat_detection_model.pkl")
data = [
    ["10.0.0.1", "GET / HTTP/1.1", "200 OK"],
    ["10.0.0.2", "POST /api/data HTTP/1.1", "404 Not Found"],
]
threats = threat_detector.detect_threats(data)
for threat in threats:
    print(threat)
