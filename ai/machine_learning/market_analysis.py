import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import os

class MarketAnalysis:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = self.load_data()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()

    def load_data(self):
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        return data

    def split_data(self):
        X = self.data.drop(['Close'], axis=1)
        y = self.data['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_random_forest(self):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        print(f'Random Forest MSE: {mse:.2f}')
        print(f'Random Forest R2: {r2:.2f}')
        return model

    def train_lstm(self):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_scaled.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train_scaled, self.y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled, self.y_test))
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        print(f'LSTM MSE: {mse:.2f}')
        print(f'LSTM R2: {r2:.2f}')
        return model

    def plot_predictions(self, model, title):
        y_pred = model.predict(self.X_test)
        plt.plot(self.y_test, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def save_model(self, model, filename):
        model.save(os.path.join('models', filename))

# Example usage
analysis = MarketAnalysis('AAPL', '2010-01-01', '2022-02-26')
rf_model = analysis.train_random_forest()
analysis.plot_predictions(rf_model, 'Random Forest Predictions')
analysis.save_model(rf_model, 'random_forest_model.h5')

lstm_model = analysis.train_lstm()
analysis.plot_predictions(lstm_model, 'LSTM Predictions')
analysis.save_model(lstm_model, 'lstm_model.h5')
