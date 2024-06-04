import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

class NodeAnomalyDetection:
    def __init__(self, node_data):
        self.node_data = node_data
        self.scaler = StandardScaler()
        self.ocsvm = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.1)

    def preprocess_data(self):
        scaled_data = self.scaler.fit_transform(self.node_data)
        return scaled_data

    def detect_anomalies(self):
        scaled_data = self.preprocess_data()
        self.ocsvm.fit(scaled_data)
        anomalies = self.ocsvm.predict(scaled_data)
        return anomalies

    def visualize_anomalies(self):
        import matplotlib.pyplot as plt

        scaled_data = self.preprocess_data()
        anomalies = self.detect_anomalies()

        plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=anomalies)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Node Anomalies")
        plt.show()

node_data = ...  # load node data
node_anomaly_detection = NodeAnomalyDetection(node_data)
anomalies = node_anomaly_detection.detect_anomalies()
node_anomaly_detection.visualize_anomalies()
