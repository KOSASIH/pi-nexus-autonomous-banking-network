import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class PiNetworkTrafficAnalyzer:
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
        # Scale the data using StandardScaler
        scaler = StandardScaler()
        self.data[self.data.columns] = scaler.fit_transform(self.data)

        # Apply PCA to reduce dimensionality
        pca = PCA(n_components=10)
        self.data[self.data.columns] = pca.fit_transform(self.data)

        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data.drop([self.target_variable], axis=1),
                                                                              self.data[self.target_variable],
                                                                              test_size=self.test_size,
                                                                              random_state=self.random_state)

    def train_models(self):
        # Create and train multiple models
        models = [
            RandomForestClassifier(n_estimators=100),
            # Add more models here
        ]

        for model in models:
            model.fit(self.X_train, self.y_train)
            self.models.append(model)

    def evaluate_models(self):
        # Evaluate each model using accuracy score, classification report, and confusion matrix
        results = []
        for model in self.models:
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred)
            matrix = confusion_matrix(self.y_test, y_pred)
            results.append((model.__class__.__name__, accuracy, report, matrix))

        return results

    def make_predictions(self, input_data):
        # Make predictions using the best model
        best_model = max(self.models, key=lambda x: x.score(self.X_test, self.y_test))
        predictions = best_model.predict(input_data)
        return predictions
