# MarketPredictionModel.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

class MarketPredictionModel:
    def __init__(self, data_file):
        self.data_file = data_file
        self.model = XGBRegressor()
        self.scaler = StandardScaler()

    def load_data(self):
        # Load historical market data
        self.data = pd.read_csv(self.data_file)
        logging.info("Data loaded successfully.")
    
    def preprocess_data(self):
        # Preprocessing: Normalize features and create target variable
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.set_index('Date', inplace=True)
        self.data['Target'] = self.data['Price'].shift(-1)  # Predict next day's price
        self.data.dropna(inplace=True)

        # Normalize features
        features = ['Open', 'High', 'Low', 'Volume']
        self.X = self.scaler.fit_transform(self.data[features])
        self.y = self.data['Target'].values

    def train_model(self):
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Hyperparameter tuning using GridSearchCV
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        grid_search = GridSearchCV(self.model, param_grid, scoring='neg_mean_squared_error', cv=3)
        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        logging.info(f"Best parameters: {grid_search.best_params_}")

        # Evaluate the model
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        logging.info(f"Model trained. MSE: {mse}, R^2: {r2}")

    def save_model(self, model_file):
        joblib.dump(self.model, model_file)
        joblib.dump(self.scaler, 'scaler.pkl')  # Save the scaler as well
        logging.info(f"Model and scaler saved to {model_file} and scaler.pkl")

    def load_model(self, model_file):
        self.model = joblib.load(model_file)
        self.scaler = joblib.load('scaler.pkl')  # Load the scaler
        logging.info(f"Model and scaler loaded from {model_file} and scaler.pkl")

    def predict(self, input_data):
        # Predict market price based on input data
        input_data_scaled = self.scaler.transform(np.array(input_data).reshape(1, -1))
        prediction = self.model.predict(input_data_scaled)
        return prediction[0]

# Example usage
if __name__ == "__main__":
    model = MarketPredictionModel("historical_market_data.csv")
    model.load_data()
    model.preprocess_data()
    model.train_model()
    model.save_model("market_prediction_model.pkl")

    # Example prediction
    input_data = [100, 105, 95, 1000]  # Open, High, Low, Volume
    predicted_price = model.predict(input_data)
    logging.info(f"Predicted next day's price: {predicted_price}")
