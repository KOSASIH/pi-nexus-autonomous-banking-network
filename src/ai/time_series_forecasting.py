import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

class TimeSeriesForecasting:
    def __init__(self, data):
        self.data = data

    def fit_model(self, order=(1, 1, 1)):
        """Fit the ARIMA model to the data."""
        self model = ARIMA(self.data, order=order)
        self.model_fit = self.model.fit()

    def forecast(self, steps=5):
        """Forecast future values."""
        forecasted_values = self.model_fit.forecast(steps=steps)
        return forecasted_values

    def plot_forecast(self, steps=5):
        """Plot the forecasted values along with the original data."""
        forecasted_values = self.forecast(steps)
        plt.figure(figsize=(12, 6))
        plt.plot(self.data, label='Historical Data')
        plt.plot(range(len(self.data), len(self.data) + steps), forecasted_values, label='Forecasted Values', color='red')
        plt.title('Time Series Forecasting')
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.legend()
        plt.show()
