import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class MarketAnalysisModel:
    def __init__(self, data):
        self.data = data
        self.scaler = MinMaxScaler()
        self.model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(self.data.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.data.drop('target', axis=1), self.data['target'], test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test))

    def predict(self, data):
        data_scaled = self.scaler.transform(data)
        return self.model.predict(data_scaled)
