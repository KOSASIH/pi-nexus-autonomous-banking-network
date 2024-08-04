import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class AnomalyDetection:
    def __init__(self, data, model_type='isolation_forest'):
        self.data = data
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()

    def preprocess_data(self):
        self.data = self.scaler.fit_transform(self.data)

    def train_isolation_forest(self):
        self.model = IsolationForest(contamination=0.1)
        self.model.fit(self.data)

    def train_one_class_svm(self):
        self.model = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.1)
        self.model.fit(self.data)

    def train_lstm_autoencoder(self):
        self.model = Sequential()
        self.model.add(LSTM(units=64, input_shape=(self.data.shape[1], 1)))
        self.model.add(Dense(self.data.shape[1]))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def train(self):
        if self.model_type == 'isolation_forest':
            self.train_isolation_forest()
        elif self.model_type == 'one_class_svm':
            self.train_one_class_svm()
        elif self.model_type == 'lstm_autoencoder':
            self.train_lstm_autoencoder()

    def predict(self, data):
        if self.model_type == 'isolation_forest':
            predictions = self.model.predict(data)
        elif self.model_type == 'one_class_svm':
            predictions = self.model.predict(data)
        elif self.model_type == 'lstm_autoencoder':
            predictions = self.model.predict(data)
        return predictions

    def evaluate(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return accuracy, precision, recall, f1

# Example usage
data = pd.read_csv('data.csv')
anomaly_detector = AnomalyDetection(data, model_type='lstm_autoencoder')
anomaly_detector.preprocess_data()
anomaly_detector.train()
predictions = anomaly_detector.predict(data)
accuracy, precision, recall, f1 = anomaly_detector.evaluate(y_true, predictions)
print(f'Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}')
