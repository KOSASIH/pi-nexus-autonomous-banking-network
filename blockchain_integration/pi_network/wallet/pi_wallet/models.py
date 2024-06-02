import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

class PortfolioModel:
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data.drop(target, axis=1), data[target], test_size=0.2, random_state=42)
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def random_forest_model(self):
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(self.X_train_scaled, self.y_train)
        y_pred = rf_model.predict(self.X_test_scaled)
        accuracy = accuracy_score(self.y_test, y_pred)
        return rf_model, accuracy

    def linear_regression_model(self):
        lr_model = LinearRegression()
        lr_model.fit(self.X_train_scaled, self.y_train)
        y_pred = lr_model.predict(self.X_test_scaled)
        mse = mean_squared_error(self.y_test, y_pred)
        return lr_model, mse

    def lstm_model(self):
        lstm_model = Sequential()
        lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(self.X_train_scaled.shape[1], 1)))
        lstm_model.add(Dense(1))
        lstm_model.compile(loss='mean_squared_error', optimizer='adam')
        lstm_model.fit(self.X_train_scaled, self.y_train, epochs=50, batch_size=32, validation_data=(self.X_test_scaled, self.y_test))
        y_pred = lstm_model.predict(self.X_test_scaled)
        mse = mean_squared_error(self.y_test, y_pred)
        return lstm_model, mse

class AIInvestmentAdvisor:
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.model = PortfolioModel(data, target)

    def get_recommendation(self):
        rf_model, accuracy = self.model.random_forest_model()
        lr_model, mse = self.model.linear_regression_model()
        lstm_model, mse_lstm = self.model.lstm_model()
        if accuracy > 0.8:
            return "Buy", rf_model
        elif mse < 0.1:
            return "Sell", lr_model
        elif mse_lstm < 0.05:
            return "Hold", lstm_model
        else:
            return "Unknown", None
