import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

class RiskManagementSystem:
    def __init__(self, data):
        self.data = data
        self.model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(self.data.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def train_model(self):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(self.data)
        X_train, y_train = self.split_data(scaled_data)
        self.model.fit(X_train, y_train, epochs=100, batch_size=32)

    def predict_risk(self, new_data):
        scaler = MinMaxScaler()
        scaled_new_data = scaler.transform(new_data)
        prediction = self.model.predict(scaled_new_data)
        return prediction

    def split_data(self, data):
        X = []
        y = []
        for i in range(len(data) - 1):
            X.append(data[i:i+1])
            y.append(data[i+1])
        return np.array(X), np.array(y)

# Example usage:
data = pd.read_csv('customer_data.csv')
risk_system = RiskManagementSystem(data)
risk_system.train_model()
new_data = pd.read_csv('new_customer_data.csv')
risk_prediction = risk_system.predict_risk(new_data)
print(risk_prediction)
