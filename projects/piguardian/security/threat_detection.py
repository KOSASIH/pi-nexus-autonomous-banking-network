import os
import json
import hashlib
import time
from collections import defaultdict
import scapy.all as scapy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class ThreatDetectionSystem:
    def __init__(self, model_file='threat_detection_model.pkl'):
        self.model_file = model_file
        self.model = self.load_model()
        self.threat_data = self.load_threat_data()

    def load_model(self):
        if os.path.exists(self.model_file):
            return RandomForestClassifier().load(self.model_file)
        else:
            return RandomForestClassifier()

    def save_model(self):
        self.model.save(self.model_file)

    def load_threat_data(self):
        if os.path.exists('threat_data.json'):
            with open('threat_data.json', 'r') as f:
                return json.load(f)
        else:
            return defaultdict(list)

    def save_threat_data(self):
        with open('threat_data.json', 'w') as f:
            json.dump(self.threat_data, f)

    def extract_features(self, packet):
        features = []
        features.append(len(packet))
        features.append(packet.haslayer(scapy.IP))
        features.append(packet.haslayer(scapy.TCP))
        features.append(packet.haslayer(scapy.UDP))
        features.append(packet.haslayer(scapy.ICMP))
        return features

    def detect_threat(self, packet):
        features = self.extract_features(packet)
        prediction = self.model.predict([features])[0]
        if prediction == 1:
            return True
        else:
            return False

    def train_model(self):
        X = []
        y = []
        for packet, label in self.threat_data.items():
            features = self.extract_features(packet)
            X.append(features)
            y.append(label)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print(f"Model accuracy: {accuracy_score(y_test, y_pred)}")
        self.save_model()

    def run(self):
        while True:
            # Sniff packets using Scapy
            packets = scapy.sniff(count=100)
            for packet in packets:
                if self.detect_threat(packet):
                    print("Threat detected!")
                    # TO DO: trigger incident response
                else:
                    print("No threat detected.")
            time.sleep(10)

if __name__ == '__main__':
    threat_detection_system = ThreatDetectionSystem()
    threat_detection_system.run()
