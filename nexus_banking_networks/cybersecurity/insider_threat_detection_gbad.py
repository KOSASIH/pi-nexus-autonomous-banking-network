# File: insider_threat_detection_gbad.py
import pandas as pd
import networkx as nx
from sklearn.ensemble import IsolationForest

class InsiderThreatDetector:
    def __init__(self, data_path):
        self.data_path = data_path
        self.graph = nx.Graph()
        self.if_model = IsolationForest(contamination=0.1)

    def build_graph(self):
        # Build graph from data
        data = pd.read_csv(self.data_path)
        for index, row in data.iterrows():
            self.graph.add_node(row['user_id'], features=row['features'])
            for neighbor in row['neighbors']:
                self.graph.add_edge(row['user_id'], neighbor)

    def detect_insider_threats(self):
        # Detect insider threats using graph-based anomaly detection
        anomalies = []
        for node in self.graph.nodes():
            features = self.graph.nodes[node]['features']
            if self.if_model.predict(features.reshape(1, -1)) == -1:
                anomalies.append(node)
        return anomalies

# Example usage:
detector = InsiderThreatDetector('data.csv')
detector.build_graph()
anomalies = detector.detect_insider_threats()
print(anomalies)
