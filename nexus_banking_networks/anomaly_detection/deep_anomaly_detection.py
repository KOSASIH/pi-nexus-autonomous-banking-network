import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class DeepAnomalyDetector:
    def __init__(self, data, seq_len=100, batch_size=32):
        self.data = data
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(LSTM(units=128, return_sequences=True, input_shape=(self.seq_len, 1)))
        model.add(LSTM(units=64))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def train(self, epochs=100):
        X_train, y_train = self.prepare_data()
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=self.batch_size, verbose=0)

    def prepare_data(self):
        data_scaled = self.scaler.fit_transform(self.data)
        X, y = [], []
        for i in range(len(data_scaled) - self.seq_len):
            X.append(data_scaled[i:i + self.seq_len])
            y.append(data_scaled[i + self.seq_len])
        X, y = np.array(X), np.array(y)
        return X.reshape(X.shape[0], X.shape[1], 1), y

    def predict(self, data):
        data_scaled = self.scaler.transform(data)
        X_pred = data_scaled[-self.seq_len:]
        X_pred = X_pred.reshape(1, X_pred.shape[0], 1)
        pred = self.model.predict(X_pred)
        return pred[0][0]

    def detect_anomalies(self, data, threshold=3):
        predictions = []
        for i in range(len(data) - self.seq_len):
            pred = self.predict(data[i:i + self.seq_len])
            predictions.append(pred)
        anomalies = [i for i, pred in enumerate(predictions) if np.abs(pred - data[i + self.seq_len]) > threshold]
        return anomalies
