import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

class RiskAssessment:
    def __init__(self, model_path='risk_model.pkl'):
        self.model_path = model_path
        self.model = None

    @staticmethod
    def load_data(file_path):
        """Load data from a CSV file."""
        data = pd.read_csv(file_path)
        return data

    @staticmethod
    def preprocess_data(data):
        """Preprocess the data for training."""
        # Example preprocessing steps
        data.fillna(0, inplace=True)  # Fill missing values
        X = data.drop('risk_label', axis=1)  # Features
        y = data['risk_label']  # Target variable
        return X, y

    def train_model(self, X, y):
        """Train the risk assessment model."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = self.model.predict(X_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        # Save the model
        joblib.dump(self.model, self.model_path)

    def load_model(self):
        """Load the trained model from a file."""
        self.model = joblib.load(self.model_path)

    def predict_risk(self, input_data):
        """Predict risk based on input data."""
        if self.model is None:
            raise Exception("Model not loaded. Please load the model first.")
        return self.model.predict(input_data)

    def evaluate_model(self, X_test, y_test):
        """Evaluate the model's performance."""
        y_pred = self.model.predict(X_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
