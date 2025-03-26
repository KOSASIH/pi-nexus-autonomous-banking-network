from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class MarketPredictor:
    """Predicts market trends using a random forest regression model.

    Attributes:
        model: The trained random forest regression model.
        data: The historical market data used to train the model.
    """

    def __init__(self, data: pd.DataFrame):
        """Creates a new market predictor with a trained random forest regression model.

        Args:
            data: The historical market data used to train the model.
        """
        self.data = data
        self.model = RandomForestRegressor(n_estimators=100, max_depth=5)
        self.model.fit(self.data.drop(["trend"], axis=1), self.data["trend"])

    def predict(self, features: pd.DataFrame) -> Optional[float]:
        """Predicts the market trend based on the given features.

        Args:
            features: The features used to predict the market trend.

        Returns:
            The predicted market trend, or None if the model could not make a prediction.
        """
        if self.model.predict(features) > 0:
            return 1.0  # Bullish trend
        else:
            return -1.0  # Bearish trend
