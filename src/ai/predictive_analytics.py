import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class PredictiveAnalytics:
    def __init__(self):
        """
        Initialize the PredictiveAnalytics class.
        """
        self.model = LinearRegression()
        self.is_trained = False

    def train(self, data, target_column):
        """
        Train the predictive model using the provided data.

        :param data: DataFrame containing the features for training.
        :param target_column: The name of the target column to predict.
        """
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate the model
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f"Model trained successfully. MSE: {mse}, R^2: {r2}")

    def predict(self, new_data):
        """
        Make predictions using the trained model.

        :param new_data: DataFrame containing the features for prediction.
        :return: Array of predictions.
        """
        if not self.is_trained:
            raise Exception("Model is not trained yet. Please train the model before making predictions.")
        
        return self.model.predict(new_data)

    def save_model(self, filename):
        """
        Save the trained model to a file.

        :param filename: The name of the file to save the model.
        """
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        """
        Load a trained model from a file.

        :param filename: The name of the file to load the model from.
        """
        self.model = joblib.load(filename)
        self.is_trained = True
        print(f"Model loaded from {filename}")

# Example usage
if __name__ == "__main__":
    # Sample data for demonstration
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 3, 4, 5, 6],
        'target': [3, 5, 7, 9, 11]
    })

    # Initialize the predictive analytics model
    analytics = PredictiveAnalytics()

    # Train the model
    analytics.train(data, target_column='target')

    # Make predictions
    new_data = pd.DataFrame({
        'feature1': [6, 7],
        'feature2': [7, 8]
    })
    predictions = analytics.predict(new_data)
    print("Predictions:", predictions)

    # Save the model
    analytics.save_model("predictive_model.pkl")

    # Load the model
    analytics.load_model("predictive_model.pkl")
