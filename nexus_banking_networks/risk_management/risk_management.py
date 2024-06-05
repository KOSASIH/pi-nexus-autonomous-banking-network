# risk_management.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class RiskManagementSystem:
    def __init__(self, data):
        self.data = data

    def train_model(self):
        X = self.data.drop(['risk_level'], axis=1)
        y = self.data['risk_level']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

    def predict_risk(self, transaction_data):
        prediction = self.model.predict(transaction_data)
        return prediction

    def evaluate_model(self):
        y_pred = self.model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

# Example usage:
data = pd.read_csv('transaction_data.csv')
rms = RiskManagementSystem(data)
rms.train_model()
transaction_data = pd.DataFrame({'amount': [1000], 'category': ['withdrawal']})
risk_level = rms.predict_risk(transaction_data)
print("Predicted Risk Level:", risk_level)
