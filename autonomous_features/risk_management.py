# risk_management.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score

class RiskManagementSystem:
    def __init__(self, transaction_data):
        self.transaction_data = transaction_data
        self.model = self.create_model()

    def create_model(self):
        model = keras.Sequential([
            keras.layers.LSTM(50, input_shape=(self.transaction_data.shape[1], 1)),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train_model(self):
        X_train, X_test, y_train, y_test = self.split_data()
        self.model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

    def predict_risk(self, new_transaction):
        new_transaction = pd.DataFrame(new_transaction, columns=self.transaction_data.columns)
        prediction = self.model.predict(new_transaction)
        return prediction[0][0]

    def split_data(self):
        X = self.transaction_data.drop('risk', axis=1)
        y = self.transaction_data['risk']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def detect_anomalies(self):
        isolation_forest = IsolationForest(contamination=0.1)
        anomalies = isolation_forest.fit_predict(self.transaction_data)
        return anomalies

# Example usage:
transaction_data = pd.read_csv('transaction_data.csv')
risk_management_system = RiskManagementSystem(transaction_data)
risk_management_system.train_model()
new_transaction = {'amount': 1000, 'category': 'withdrawal', 'location': 'New York'}
risk_score = risk_management_system.predict_risk(new_transaction)
print(f'Risk score: {risk_score:.2f}')
