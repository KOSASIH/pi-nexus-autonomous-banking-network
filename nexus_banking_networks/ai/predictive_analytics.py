import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class PredictiveAnalytics:
    def __init__(self, data):
        self.data = data

    def train_model(self):
        # Train random forest regressor model on historical data
        X = self.data.drop(['target'], axis=1)
        y = self.data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    def make_predictions(self, model):
        # Make predictions on new data using trained model
        predictions = model.predict(self.data)
        return predictions
