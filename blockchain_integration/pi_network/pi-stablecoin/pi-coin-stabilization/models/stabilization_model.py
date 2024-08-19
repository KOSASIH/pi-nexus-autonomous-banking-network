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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pykalman import KalmanFilter

class StabilizationModel:
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
            MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1000),
            self._train_lstm_model(),
            self._train_arima_model(),
            self._train_sarimax_model(),
            self._train_kalman_filter_model()
        ]

        for model in models:
            model.fit(self.X_train, self.y_train)
            self.models.append(model)

    def _train_lstm_model(self):
        # Train an LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(self.X_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def _train_arima_model(self):
        # Train an ARIMA model
        model = ARIMA(self.y_train, order=(1,1,1))
        model_fit = model.fit(disp=0)
        return model_fit

    def _train_sarimax_model(self):
        # Train a SARIMAX model
        model = SARIMAX(self.y_train, order=(1,1,1), seasonal_order=(1,1,1,12))
        model_fit = model.fit(disp=0)
        return model_fit

    def _train_kalman_filter_model(self):
        # Train a Kalman filter model
        kf = KalmanFilter(transition_matrices=[1],
                          observation_matrices=[1],
                          initial_state_mean=0,
                          initial_state_covariance=1,
                          observation_covariance=1,
                          transition_covariance=0.1)
        kf = kf.em(self.y_train, n_iter=50)
        return kf

    def evaluate_models(self):
        # Evaluate each model using mean squared error and R-squared score
        results = []
        for model in self.models:
            y_pred = model.predict(self.X_test)
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            results.append((model.__class__.__name__, mse, r2))

        return results

    def make_predictions(self, input
