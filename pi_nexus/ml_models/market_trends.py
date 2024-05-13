import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential


class MarketTrendsModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.target = None
        self.scaler = None

    def load_data(self):
        self.data = pd.read_csv(self.data_path)
        self.target = self.data["target"]
        self.data = self.data.drop("target", axis=1)

    def preprocess_data(self):
        # Perform data preprocessing steps such as missing value imputation, feature scaling, etc.
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.data)

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.target, test_size=0.2, random_state=42
        )

    def build_lstm_model(self):
        model = Sequential()
        model.add(
            LSTM(units=50, activation="relu", input_shape=(self.X_train.shape[1], 1))
        )
        model.add(Dropout(0.2))
        model.add(Dense(units=25, activation="relu"))
        model.add(Dense(units=1))
        model.compile(optimizer="adam", loss="mse")
        return model

    def train_lstm_model(self, model):
        model.fit(
            self.X_train,
            self.y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            verbose=1,
        )

    def evaluate_lstm_model(self, model):
        y_pred = model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        print(f"MSE: {mse}, R2: {r2}")

    def run(self):
        self.load_data()
        self.preprocess_data()
        self.split_data()
        model = self.build_lstm_model()
        self.train_lstm_model(model)
        self.evaluate_lstm_model(model)


if __name__ == "__main__":
    model = MarketTrendsModel("data/market_data.csv")
    model.run()
