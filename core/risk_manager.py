import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class RiskManager:
    def __init__(self, data):
        self.data = data
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train_model(self):
        X = self.data.drop(['risk_level'], axis=1)
        y = self.data['risk_level']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

    def predict_risk(self, transaction):
        features = pd.DataFrame([transaction], columns=self.data.columns[:-1])
        prediction = self.model.predict(features)
        return prediction[0]

    def update_model(self, new_data):
        self.data = pd.concat([self.data, new_data])
        self.train_model()

# Example usage:
risk_manager = RiskManager(pd.read_csv('risk_data.csv'))
risk_manager.train_model()

transaction = {'amount': 1000, 'country': 'USA', 'card_type': 'VISA'}
risk_level = risk_manager.predict_risk(transaction)
print(f'Predicted risk level: {risk_level}')
