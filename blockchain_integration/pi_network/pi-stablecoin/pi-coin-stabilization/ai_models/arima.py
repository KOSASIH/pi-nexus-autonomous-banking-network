import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

class ARIMAModel:
    def __init__(self, data, order=(5,1,0)):
        self.data = data
        self.order = order
        self.model = None

    def train_model(self):
        # Create ARIMA model
        model = ARIMA(self.data, order=self.order)

        # Train model
        model_fit = model.fit()

        # Set model
        self.model = model_fit

    def evaluate_model(self):
        # Make predictions on entire dataset
        predictions = self.model.predict(start=0, end=len(self.data))

        # Calculate mean squared error
        mse = mean_squared_error(self.data, predictions)

        return mse

    def make_predictions(self, steps):
        # Make predictions for future steps
        predictions = self.model.predict(start=len(self.data), end=len(self.data)+steps)

        return predictions
