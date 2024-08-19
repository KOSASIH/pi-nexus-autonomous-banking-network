import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class DecisionTreeModel:
    def __init__(self, data, target_variable, test_size=0.2, random_state=42, max_depth=5):
        self.data = data
        self.target_variable = target_variable
        self.test_size = test_size
        self.random_state = random_state
        self.max_depth = max_depth
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def preprocess_data(self):
        # Drop missing values
        self.data.dropna(inplace=True)

        # Scale features using StandardScaler
        scaler = StandardScaler()
        self.data[['feature1', 'feature2', ...]] = scaler.fit_transform(self.data[['feature1', 'feature2', ...]])

        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data.drop([self.target_variable], axis=1),
                                                                              self.data[self.target_variable],
                                                                              test_size=self.test_size,
                                                                              random_state=self.random_state)

    def train_model(self):
        # Create pipeline with decision tree regressor
        pipeline = Pipeline([
            ('decision_tree', DecisionTreeRegressor(max_depth=self.max_depth))
        ])

        # Train model
        pipeline.fit(self.X_train, self.y_train)

        # Set model
        self.model = pipeline

    def evaluate_model(self):
        # Make predictions on test set
        y_pred = self.model.predict(self.X_test)

        # Calculate mean squared error
        mse = mean_squared_error(self.y_test, y_pred)

        # Calculate R-squared score
        r2 = r2_score(self.y_test, y_pred)

        return mse, r2

    def make_predictions(self, input_features):
        # Make predictions using trained model
        predictions = self.model.predict(input_features)

        return predictions
