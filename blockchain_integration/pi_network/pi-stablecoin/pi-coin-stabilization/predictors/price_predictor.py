import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

class PricePredictor:
    def __init__(self, data, target_variable, test_size=0.2, random_state=42):
        self.data = data
        self.target_variable = target_variable
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = []

    def preprocess_data(self):
        # Drop missing values
        self.data.dropna(inplace=True)

        # Scale features using StandardScaler
        scaler = StandardScaler()
        self.data[['feature1', 'feature2', ...]] = scaler.fit_transform(self.data[['feature1', 'feature2', ...]])

        # Select top k features using SelectKBest
        selector = SelectKBest(f_classif, k=10)
        self.data = selector.fit_transform(self.data.drop([self.target_variable], axis=1), self.data[self.target_variable])

        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data.drop([self.target_variable], axis=1),
                                                                              self.data[self.target_variable],
                                                                              test_size=self.test_size,
                                                                              random_state=self.random_state)

    def train_models(self):
        # Create and train multiple models
        models = [
            RandomForestRegressor(n_estimators=100, max_depth=5),
            XGBRegressor(objective='reg:squarederror', max_depth=5, n_estimators=100),
            CatBoostRegressor(iterations=100, depth=5, learning_rate=0.1),
            LGBMRegressor(objective='regression', max_depth=5, n_estimators=100),
            LinearRegression(),
            SVR(kernel='rbf', C=1e3, gamma=0.1),
            MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1000)
        ]

        for model in models:
            model.fit(self.X_train, self.y_train)
            self.models.append(model)

    def evaluate_models(self):
        # Evaluate each model using mean squared error and R-squared score
        results = []
        for model in self.models:
            y_pred = model.predict(self.X_test)
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            results.append((model.__class__.__name__, mse, r2))

        return results

    def make_predictions(self, input_features):
        # Make predictions using the best model
        best_model = max(self.models, key=lambda x: x.score(self.X_test, self.y_test))
        predictions = best_model.predict(input_features)

        return predictions
