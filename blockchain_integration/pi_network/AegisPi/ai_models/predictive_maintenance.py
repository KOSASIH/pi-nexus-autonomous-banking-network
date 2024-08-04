import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class PredictiveMaintenance:
    def __init__(self, data, target_variable, model_type='random_forest'):
        self.data = data
        self.target_variable = target_variable
        self.model_type = model_type
        self.model = None

    def preprocess_data(self):
        self.data = self.data.dropna()
        self.data = self.data.drop_duplicates()

    def train_random_forest(self):
        X = self.data.drop(self.target_variable, axis=1)
        y = self.data[self.target_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

    def train_lstm(self):
        X = self.data.drop(self.target_variable, axis=1)
        y = self.data[self.target_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = Sequential()
        self.model.add(LSTM(units=64, input_shape=(X.shape[1], 1)))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    def train(self):
        if self.model_type == 'random_forest':
            self.train_random_forest()
        elif self.model_type == 'lstm':
            self.train_lstm()

    def predict(self, data):
        if self.model_type == 'random_forest':
            predictions = self.model.predict(data)
        elif self.model_type == 'lstm':
            predictions = self.model.predict(data)
        return predictions

    def evaluate(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        return mse, mae

# Example usage
data = pd.read_csv('data.csv')
target_variable = 'emaining_useful_life'
predictive_maintenance = PredictiveMaintenance(data, target_variable, model_type='lstm')
predictive_maintenance.preprocess_data()
predictive_maintenance.train()
predictions = predictive_maintenance.predict(data.drop(target_variable, axis=1))
mse, mae = predictive_maintenance.evaluate(data[target_variable], predictions)
print(f'MSE: {mse:.3f}, MAE: {mae:.3f}')
