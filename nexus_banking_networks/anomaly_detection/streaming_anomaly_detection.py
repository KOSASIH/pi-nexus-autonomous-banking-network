import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from kafka import KafkaConsumer
from kafka import KafkaProducer

class StreamingAnomalyDetector:
    def __init__(self, kafka_topic, bootstrap_servers, seq_len=100):
        self.kafka_topic = kafka_topic
        self.bootstrap_servers = bootstrap_servers
        self.seq_len = seq_len
        self.scaler = StandardScaler()
        self.consumer = KafkaConsumer(self.kafka_topic, bootstrap_servers=self.bootstrap_servers)
        self.producer = KafkaProducer(bootstrap_servers=self.bootstrap_servers)

    def detect_anomalies(self, data):
        data_scaled = self.scaler.transform(data)
        X_pred = data_scaled[-self.seq_len:]
        X_pred = X_pred.reshape(1, X_pred.shape[0], 1)
        pred = self.model.predict(X_pred)
        anomaly_score = np.abs(pred - data_scaled[-1])
        if anomaly_score > 3:
            self.producer.send(self.kafka_topic, value=b'anomaly_detected')
        else:
            self.producer.send(self.kafka_topic, value=b'no_anomaly')

    def start_streaming(self):
        for message in self.consumer:
            data = pd.read_csv(message.value, header=None)
            self.detect_anomalies(data)
