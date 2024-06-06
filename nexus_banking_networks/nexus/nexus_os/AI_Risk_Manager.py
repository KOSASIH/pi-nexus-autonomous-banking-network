import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class AIRiskManager:
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = RandomForestClassifier(n_estimators=100)

    def train_model(self):
        X = self.dataset.drop(['risk_level'], axis=1)
        y = self.dataset['risk_level']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

    def predict_risk(self, transaction_data):
        prediction = self.model.predict(transaction_data)
        return prediction

    def update_model(self, new_data):
        self.dataset = pd.concat([self.dataset, new_data])
        self.train_model()

# Example usage:
risk_manager = AIRiskManager(pd.read_csv('risk_data.csv'))
risk_manager.train_model()

# Predict risk level for a new transaction
transaction_data = pd.DataFrame({'amount': [1000], 'category': ['withdrawal']})
risk_level = risk_manager.predict_risk(transaction_data)
print(f'Predicted risk level: {risk_level}')
