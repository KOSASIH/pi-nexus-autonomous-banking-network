import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class MarketPredictionModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print("Market Prediction Model Training Report:")
        print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
        print("R2 Score:", r2_score(y_test, y_pred))

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        y_pred = self.model.predict(X)
        print("Market Prediction Model Evaluation Report:")
        print("Mean Squared Error:", mean_squared_error(y, y_pred))
        print("R2 Score:", r2_score(y, y_pred))

# Example usage:
if __name__ == "__main__":
    # Load market data
    market_data = pd.read_csv("../data/market-data.csv")

    # Preprocess data
    X = market_data.drop(["asset_id", "timestamp"], axis=1)
    y = market_data["price"]

    # Create and train model
    model = MarketPredictionModel()
    model.train(X, y)

    # Evaluate model
    model.evaluate(X, y)
