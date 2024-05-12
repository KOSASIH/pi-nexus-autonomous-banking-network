import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

class TimeSeriesAnalysis:
    def __init__(self, data, target_column, exogenous_variables):
        self.data = data
        self.target_column = target_column
        self.exogenous_variables = exogenous_variables

    def fit_model(self):
        """
        Fits the time series analysis model to the data.
        """
        X = self.data[self.exogenous_variables]
        y = self.data[self.target_column]

        model = SARIMAX(y, exog=X, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit()

        return model_fit

    def predict_market_trends(self, num_periods):
        """
        Predicts market trends using the time series analysis model.
        """
        model_fit = self.fit_model()
        forecast = model_fit.forecast(steps=num_periods, exog=self.data[self.exogenous_variables].tail(num_periods))

        return forecast
